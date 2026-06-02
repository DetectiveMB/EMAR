from torch import concat, einsum, nn, mm, stack, tensor, Tensor
from torch.nn.functional import relu, log_softmax, softmax

class MultipleRankingLoss(nn.Module):
    """
    Multiple ranking Loss function with classification.
    """

    def __init__(self, device, temperature=1):
        super(MultipleRankingLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.temperature = temperature
        
        self.device = device
    def forward(
        self,
        model_output
    ) -> Tensor:
        anchors, positives = model_output[0], model_output[1]
        ce_similarity = mm(anchors / self.temperature, positives.T)
        labels = tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        ce_loss = self.CELoss(ce_similarity, labels)

        accuracy = (ce_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)
        
        return ce_loss, accuracy



class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, output) -> Tensor:
        pos_scores = einsum('ij,ij->i', output['Q_emb'], output['P_emb']) #output['Q_emb'] @ output['P_emb'].T
        neg_scores = einsum('ij,ij->i', output['Q_emb'], output['N_emb']) #output['Q_emb'] @ output['N_emb'].T
        hard_loss = relu(self.margin - pos_scores + neg_scores).mean()
        accuracy = pos_scores > neg_scores

        # only you can have a batch with more than 1 element do random negative batching:
        
        if output['P_emb'].shape[0] > 1:
            repeated_pos_scores = pos_scores.view(-1,1).repeat_interleave(
                (output['P_emb'].shape[0] - 1)*2, dim=1
            )
            query_pos_scores = output['Q_emb'] @ output['P_emb'].T # batch_size * batch_size
            
            masks = [[j for j in range(output['Q_emb'].shape[0]) if j != i] for i in range(output['Q_emb'].shape[0])]
            neg_docs_score_1 = query_pos_scores.gather(1, tensor(masks).to(query_pos_scores.device))

            query_neg_scores = output['Q_emb'] @ output['N_emb'].T # batch_size * batch_size
            neg_docs_score_2 = query_neg_scores.gather(1, tensor(masks).to(query_neg_scores.device))

            neg_docs_score = concat((neg_docs_score_1, neg_docs_score_2), dim=1)
            
            in_batch_loss = relu(self.margin - repeated_pos_scores + neg_docs_score).mean()
            hard_loss = hard_loss + in_batch_loss
        
        
        
        return hard_loss, accuracy
    

class MoEDistillationTripleRankingLoss(nn.Module):
    """
    Triplet Margin Loss function with MoE distillation.
    """

    def __init__(self, margin=1.0):
        super(MoEDistillationTripleRankingLoss, self).__init__()
        self.margin = margin
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output, labels) -> Tensor:

        pos_scores = einsum('ij,ij->i', output['Q_emb'], output['P_emb'])
        neg_scores = einsum('ij,ij->i', output['Q_emb'], output['N_emb'])
        
        scores = concat((pos_scores, neg_scores))
        labels_scores = concat((labels['P_scores'],labels['N_scores'])).to(pos_scores.device)
        
        kd_loss = self.l1_loss(scores, labels_scores)

        ranking_accuracy = pos_scores > neg_scores

        # cross entropy for the MoE
        ce_q_loss = self.ce_loss(output['Q_gate'], labels['domain'].to(output['Q_gate'].device)) 
        ce_p_loss = self.ce_loss(output['P_gate'], labels['domain'].to(output['P_gate'].device)) 
        ce_n_loss = self.ce_loss(output['N_gate'], labels['domain'].to(output['N_gate'].device))
        ce_loss = ce_q_loss + ce_p_loss + ce_n_loss
        
        gate_accuracy = concat(
            (
                output['Q_gate'].argmax(dim=1, keepdim=True).squeeze().cpu() == labels['domain'].cpu(),
                output['P_gate'].argmax(dim=1, keepdim=True).squeeze().cpu() == labels['domain'].cpu(),
                output['N_gate'].argmax(dim=1, keepdim=True).squeeze().cpu() == labels['domain'].cpu()
            )
        )
        return kd_loss, ce_loss, ranking_accuracy, gate_accuracy
    

class DistillationTripleRankingLoss(nn.Module):
    """
    Triplet Margin Loss function with MoE distillation.
    """

    def __init__(self, margin=1.0):
        super(DistillationTripleRankingLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = TripletMarginLoss(self.margin)
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, output, labels) -> Tensor:

        pos_scores = einsum('ij,ij->i', output['Q_emb'], output['P_emb'])
        neg_scores = einsum('ij,ij->i', output['Q_emb'], output['N_emb'])

        scores = concat((pos_scores, neg_scores))
        
        labels_scores = concat((labels['P_scores'], tensor(labels['N_scores']))).to(pos_scores.device).float()

        kd_loss_1 = self.mse_loss(scores, labels_scores)

        hard_loss = relu(
            self.margin - pos_scores + neg_scores
        ).mean()
        ranking_accuracy = pos_scores > neg_scores

        
        kd_loss = kd_loss_1 + hard_loss #kd_loss_1 + kd_loss_2 # + hard_loss
        
        return kd_loss, ranking_accuracy
        
    

class MoEGateCrossEntropyLoss(nn.Module):
    """
    Triplet Margin Loss function with MoE distillation.
    """

    def __init__(self, num_classes):
        super(MoEGateCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.num_classes = num_classes

    def forward(self, output, labels) -> Tensor:

        # cross entropy for the MoE
        gate_output = output['gate_output']
        
        domain_labels = labels['domain'].to(output['gate_output'].device)
        
        # ce_q_loss = self.ce_loss(gate_output, domain_labels) 
        
        labels_one_hot = nn.functional.one_hot(domain_labels, num_classes=self.num_classes).float()
        ce_loss = self.ce_loss(gate_output, labels_one_hot) # ce_q_loss
        
        gate_accuracy = gate_output.argmax(dim=1, keepdim=True).squeeze().cpu() == domain_labels.cpu()
        
        return ce_loss, gate_accuracy
    

class CrossEncoderLoss(nn.Module):
    """
    Multiple ranking Loss function with classification.
    """

    def __init__(self):
        super(CrossEncoderLoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        model_output
    ) -> Tensor:
        pos_score = model_output['P_score']
        pos_label = tensor([1 for _ in range(pos_score.shape[0])], device=pos_score.device).unsqueeze(1).float()

        neg_score = model_output['N_score']
        neg_label = tensor([0 for _ in range(neg_score.shape[0])], device=neg_score.device).unsqueeze(1).float()

        labels = concat((pos_label, neg_label))
        scores = concat((pos_score, neg_score))

        ce_loss = self.BCELoss(scores, labels)

        accuracy = ((scores > 0.5).float() == labels).cpu()
        return ce_loss, accuracy



class DistillationCrossEncoderLoss(nn.Module):
    """
    Triplet Margin Loss function with MoE distillation.
    """

    def __init__(self):
        super(DistillationCrossEncoderLoss, self).__init__()
        self.ce_loss = CrossEncoderLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, output, labels) -> Tensor:


        pos_scores = output['P_score']
        neg_scores = output['N_score']
        
        scores = concat((pos_scores, neg_scores))
        
        labels_scores = concat((labels['P_scores'], labels['N_scores'])).to(pos_scores.device).float().view(-1,1)

        kd_loss_1 = self.mse_loss(scores, labels_scores)

        hard_loss, ranking_accuracy = self.ce_loss(output)
        
        kd_loss = hard_loss + kd_loss_1
        
        return kd_loss, ranking_accuracy
