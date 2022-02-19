import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10

class PSNR_RGB(nn.Module):
    def __init__(self, max_val):
        super(PSNR_RGB, self).__init__()

    def __call__(self, a, b):
        mse = torch.mean((a.float()-b.float())**2)

        if mse == 0:
            return torch.tensor(0)

        psnr = 10*torch.log10(255*255 / mse)

        return psnr

class PSNR_YCbcr(nn.Module):
    def __init__(self):
        super(PSNR_YCbcr, self).__init__()

    def __call__(self, a, b):
        a = a.float()[0]
        b = b.float()[0]
        Y_a = 0.256789*a[...,0] + 0.504129*a[...,1] + 0.097906*a[...,2] + 16
        Y_b = 0.256789*b[...,0] + 0.504129*b[...,1] + 0.097906*b[...,2] + 16

        mse = torch.mean((Y_a-Y_b)**2)
        if mse == 0:
            return torch.tensor(0)

        psnr = 10*torch.log10(255*255/mse)

        return psnr