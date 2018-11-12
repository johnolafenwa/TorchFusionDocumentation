
Training With Custom Datasets!
=================
While TorchFusion provides pre-defined loaders for popular standard datasets. Very often, you will need to train on your own custom datasets.
TorcFusion provides loaders for any image dataset organized into a single folder with subfolders representing each class of images. For example,
if you are training a model to recognize cats and dogs, you should have a train folder with two subfolders, one for dogs and one for cats. 

**Import Classes** ::

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


**Load the dataset** ::
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_loader = imagefolder_loader(transform=train_transforms,batch_size=32,shuffle=True,root="path-to-train-folder")
    test_loader = imagefolder_loader(transform=test_tranforms,shuffle=False,batch_size=32,root="path-to-test-folder")

Data augmentation helps to improve the performance of our models, hence,  we overrided the default transformations of
torchfusion with a new one containing our custom transforms.

**Define the model** ::

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

To make the code more compact above, we first defined a `Unit` module that we reused in the model. Notice how we initialized
the convolution layer with `Kaiming Normal` in the above, all torchfusion convolution layers are by default initialized
with `Kaiming_Normal` and all Linear layers have default init of `Xavier_Normal`, however, we explicitly defined the intialization
here to demonstrate how you can use any of the many initializers that torchfusion provides to initialize your layers.
The `bias_init` arguement also allows you to initialize the bias as you want.


**Define optimizer, lr scheduler and loss** ::

    if cuda.is_available():
        model = model.cuda()
    
    optimizer = Adam(model.parameters(),lr=0.001)

    lr_scheduler = StepLR(optimizer,step_size=30,gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    train_metrics = [Accuracy()]
    test_metrics = [Accuracy()]

In the above, we defined a learning rate scheduler to reduce the learning rate by a factor of 10 every 30 epochs.
There are many learning rate schedulers in pyorch's lr_scheduler package, you can use any of them here.


**Train the model** ::

    learner = StandardLearner(model)

    if __name__ == "__main__":
        learner.train(train_loader,train_metrics=train_metrics,optimizer=optimizer,loss_fn=loss_fn,model_dir="./custom-models",test_loader=test_loader,test_metrics=test_metrics,num_epochs=200,batch_log=False,lr_scheduler=lr_scheduler,save_logs="custom-model-logs.txt",display_metrics=True,save_metrics=True)

Here we specified a number of additional arguements, first we specified the `lr_scheduler` we earlier created,
next we specified `save_logs`, this will save all logs to the file we specified, finally, `save_metrics` and `display_metrics` will
display visualization of loss and metrics and save the generated plots.
The save plots,logs and models can all be found in the directory `custom-models` that we specified above.

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
        transforms.RandomCrop(224,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_loader = imagefolder_loader(transform=train_transforms,batch_size=32,shuffle=True,root="path-to-train-folder")
    test_loader = imagefolder_loader(transform=test_tranforms,shuffle=False,batch_size=32,root="path-to-test-folder")

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
    
    optimizer = Adam(model.parameters(),lr=0.001)

    lr_scheduler = StepLR(optimizer,step_size=30,gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    train_metrics = [Accuracy()]
    test_metrics = [Accuracy()]

    learner = StandardLearner(model)

    learner = StandardLearner(model)

    if __name__ == "__main__":
        learner.train(train_loader,train_metrics=train_metrics,optimizer=optimizer,loss_fn=loss_fn,model_dir="./cifar10-models",test_loader=test_loader,test_metrics=test_metrics,num_epochs=30,batch_log=False,lr_scheduler=lr_scheduler,save_logs="cifar10-logs.txt",display_metrics=True,save_metrics=True)











