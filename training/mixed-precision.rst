Mixed Precision Training
=========================
Deep Learning models are usually trained using standard 32 bit floating point arithmetic. To speed up the training of deep learning models, a lot of research has gone into using lower bit precison arithmetic such as 8 bit and 16 bit arithmetic. The lower bits are much faster than the 32 bit precision arithmetic. NVIDIA Volta GPUs have tensor cores specialized for 16 bit precison arithmetic, taking advantage of them could lead to significant speed up in training of large deep learning models.

To learn more about mixed precision training, read ``Nvidia Mixed Precision Training  https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html/`_

The greatest challenge with training in mixed precision made is radical decrease in model accuracy. Hence, a naive implementation of mixed precision training will result in very poor performance. TorchFusion includes highly optimized procedures for training Deep Learning Models in mixed precision without compromising performance.

**PROCEDURE**

The following are the procedures you need to follow to train torchfusion models in mixed precision.


**Import Classes** ::

    from torchfusion.fp16_utils import half_model, FP16_Optimizer
    
**CONVERT MODEL AND OPTIMIZER INTO FP16** ::

    model = half_model(model)
    optimizer = FP16_Optimizer(Adam(model.parameters()))
    learner = StandardLearner(model)
    learner.half()
    
    # if using lr_scheduler
    lr_scheduler = StepLR(optimizer.optimizer,step_size=30,gamma=0.1)
    


**PUTTING IT ALL TOGETHER** ::

   
    from torchfusion.layers import *
    from torchfusion.datasets import *
    from torchfusion.metrics import *
    from torchfusion.initializers import Kaiming_Normal, Xavier_Normal
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.cuda as cuda
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from torchfusion.learners import StandardLearner

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    test_transforms = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_loader = cifar10_loader(transform=train_transforms,batch_size=32)
    test_loader = cifar10_loader(transform=test_transforms,train=False,batch_size=32)

    class Unit(nn.Module):
        def __init__(self,in_channels,out_channels):
            super(Unit,self).__init__()
            self.conv = Conv2d(in_channels,out_channels,kernel_size=3,padding=1,weight_init=Kaiming_Normal())
            self.bn = BatchNorm2d(out_channels)
            self.activation = Swish()

        def forward(self,inputs):
            outputs = self.conv(inputs)
            outputs = self.bn(outputs)
            return self.activation(outputs)

    model = nn.Sequential(
        Unit(3,64),
        Unit(64,64),
        Unit(64,64),
        nn.Dropout(0.25),

        nn.MaxPool2d(kernel_size=3,stride=2),

        Unit(64,128),
        Unit(128,128),
        Unit(128,128),
        nn.Dropout(0.25),

        nn.MaxPool2d(kernel_size=3,stride=2),

        Unit(128,256),
        Unit(256,256),
        Unit(256,256),

        GlobalAvgPool2d(),

        Linear(256, 10,weight_init=Xavier_Normal())
    )


    if cuda.is_available():
        model = model.cuda()
    model = half_model(model)
    
    optimizer = FP16_Optimizer(Adam(model.parameters(),lr=0.001))

    lr_scheduler = StepLR(optimizer.optimizer,step_size=30,gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    train_metrics = [Accuracy()]
    test_metrics = [Accuracy()]

    learner = StandardLearner(model)
    learner.half()

    if __name__ == "__main__":
        learner.train(train_loader,train_metrics=train_metrics,optimizer=optimizer,loss_fn=loss_fn,model_dir="./cifar10-models",test_loader=test_loader,test_metrics=test_metrics,num_epochs=30,batch_log=False,lr_scheduler=lr_scheduler,save_logs="cifar10-logs.txt",display_metrics=True,save_metrics=True)











