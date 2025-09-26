# Usage:
Very simple torch module. Import it as ``from Tournaments import Tournament``.
To use, define the number of outputs (classes) during initialization.

Forward takes ``<tournament>.num_edges`` inputs. You can define this module first and use that variable to define the output dimension of the previous layer.

Best practice is to do softmax and use symmetric cross entropy (sce) loss:

$\mathcal{L} = -\sum_{k=1}^{K} \left[ y_k \log(p_k) + (1 - y_k) \log(1 - p_k) \right]$,

``loss = -(targets * torch.log(softmax_preds + 1e-10) + (1 - targets) * torch.log(1 - softmax_preds + 1e-10))``

where targets $y_k$ are one-hot vectors (can be soft targets too!) to train, and **no softmax** under inference. Just do argmax or analyze confidences directly.
I have included this loss in ``Tournament.py`` as well, so you can import it with ``from Tournaments import symmetric_cross_entropy`` and pass it preds and targets.

Note that the inputs to the tournament should be between 0 and 1, and how you do that is up to you. I would recommend ``sigmoid``$\rightarrow$``tournament``$\rightarrow$``softmax``$\rightarrow$``sce`` for training and ``sigmoid``$\rightarrow$``tournament`` for inference.

Good luck, please cite me, I will update the arxiv link soon <3!
