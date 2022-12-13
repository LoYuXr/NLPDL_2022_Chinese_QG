import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration
from typing import Optional, Tuple
class AdvContrastiveQG(nn.Module):
    def __init__(self, config, model_args, data_args, training_args, device):
        super(AdvContrastiveQG, self).__init__()
        
        ## adversarial contrastive learning use
        self.config = config
        self.tau = model_args.tau
        self.adv = model_args.adv  #true or false
        self.pos_eps = model_args.pos_eps
        self.neg_eps = model_args.neg_eps
        self.device = device
        self.counter = 0
        self.first_eval = True
        self.counter_thresh = 2000000
        self.maxcounter = 4000000
        ##notice, model_args need to add those and the projection layer!
        
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.main_input_name = self.mt5.main_input_name
        self.generate = self.mt5.generate
        self.config.pad_token_id = self.mt5.config.pad_token_id
        self.projection = nn.Sequential(nn.Linear(model_args.hidden_size, model_args.hidden_size),
                                        nn.ReLU())
        self.encoder = self.mt5.encoder
        self.decoder = self.mt5.decoder
        self.model_parallel = self.mt5.model_parallel
        self.config = self.mt5.config
        self.lm_head = self.mt5.lm_head
        self._shift_right = self.mt5._shift_right
        
        #### dont know hidden_size of encoder_output[0]
        
    def resize_token_embeddings(self, tokenizer_length):
        return self.mt5.resize_token_embeddings(tokenizer_length)
        
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        ##caution
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
                
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        vocab_size = lm_logits.size(-1)
        
        #compute loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
        if lm_logits.requires_grad == False and self.first_eval == True:
           ## eval/predict
           self.first_eval = False
           self.maxcounter = 2*self.counter
           self.counter_thresh = 4*self.counter
           
        # implement contrastive adversarial training:
        if self.adv and lm_logits.requires_grad == True and self.counter >= self.counter_thresh:
            
            proj_enc_h = self.projection(hidden_states)
            proj_dec_h = self.projection(sequence_output)
            avg_doc = self.avg_pool(proj_enc_h, attention_mask)
            avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)
            
            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_doc.unsqueeze(1), avg_abs.unsqueeze(0))

            # decoder
            perturbed_dec = self.generate_adv(sequence_output,labels)
            batch_size = input_ids.size(0)
            
            proj_pert_dec_h = self.projection(perturbed_dec)
            avg_pert = self.avg_pool(proj_pert_dec_h, decoder_attention_mask)
            
            adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)
            
            pos_dec_hidden, kl_loss = self.generate_cont_adv(hidden_states, attention_mask,
                                                    sequence_output, decoder_attention_mask,
                                                    lm_logits,
                                                    self.tau, self.pos_eps)
            
            avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
                                        decoder_attention_mask)

            pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)
            logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            identity = torch.eye(batch_size, device=input_ids.device)
            pos_sim = identity * pos_sim
            neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            new_sim_matrix = pos_sim + neg_sim
            new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

            labels = torch.arange(batch_size,
                                  device=input_ids.device)

            cont_loss = cont_crit(logits, labels)
            new_cont_loss = cont_crit(new_logits, labels)
            
            cont_loss = min(1, (self.counter-self.counter_thresh)/self.maxcounter) * (cont_loss + new_cont_loss)
            
            loss += cont_loss
            loss -= min(1, (self.counter-self.counter_thresh)/self.maxcounter) * kl_loss
            
        self.counter +=1
            
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]# + encoder_outputs
            
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
       
        
    def generate_adv(self, dec_hiddens, labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.lm_head(dec_hiddens)
        
        criterion = CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         labels.view(-1))
        
        loss.backward()
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]
        self.zero_grad()

        return perturbed_dec

    
    def generate_cont_adv(self, enc_hiddens, enc_mask, dec_hiddens, dec_mask, lm_logits, tau, eps):
        
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.projection(enc_hiddens), enc_mask)

        avg_dec = self.avg_pool(self.projection(dec_hiddens), dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0), device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.lm_head(perturb_dec_hidden)
        
        true_probs = F.softmax(lm_logits, -1)  #(8,100,250112)
        
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl_loss = Variable(kl, requires_grad = True)
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

        return perturb_dec_hidden, kl_loss

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
    
    def process_batch(self, batch):
        (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels) = batch
        src_length = torch.sum(attention_mask, 1)
        max_src_length = torch.max(src_length)

        trg_length = torch.sum(decoder_attention_mask, 1)
        max_trg_length = torch.max(trg_length)

        input_ids = input_ids[:, :max_src_length]
        attention_mask = attention_mask[:, :max_src_length]

        decoder_input_ids = decoder_input_ids[:, :max_trg_length]
        decoder_attention_mask = decoder_attention_mask[:, :max_trg_length]
        labels = labels[:, :max_trg_length]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels
        }
        return inputs