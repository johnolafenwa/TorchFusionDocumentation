
Buiding Custom Trainers!
========================
Torchfusion provides a wide variety of GAN Learners, you will find them in the torchfusion.gan.learners package
However, lots of research is ongoing into improved techniques for GANs, hence, we provide multiple levels of abstractions
to faciliate research.


**Custom Loss** ::

    #Extend the StandardBaseGanLearner
    class CustomGanLearner(StandardBaseGanLearner):
        #Override the __update_discriminator_loss__
        def __update_discriminator_loss__(self, real_images, gen_images, real_preds, gen_preds):

            pred_loss = -torch.mean(real_preds - gen_preds)

            return pred_loss
    
        #Override the __update_generator_loss__
        def __update_generator_loss__(self,real_images,gen_images,real_preds,gen_preds):

            pred_loss = -torch.mean(gen_preds - real_preds)
            return pred_loss


**Custom Training Logic** ::

    #Extend BaseGanCore
    class CustomGanLearner(BaseGanCore):

        #Extend train 
        def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,loss_fn=nn.BCELoss(),**kwargs):

            self.latent_size = latent_size
            self.loss_fn = loss_fn
            super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,**kwargs)
        
        #Extend __disc_train_func__
        def __disc_train_func__(self, data):

            super().__disc_train_func__(data)

            self.disc_optimizer.zero_grad()

            if isinstance(data, list) or isinstance(data, tuple):
                x = data[0]
            else:
                x = data

            batch_size = x.size(0)

            source = self.dist.sample((batch_size,self.latent_size))

            real_labels = torch.ones(batch_size,1)
            fake_labels = torch.zeros(batch_size,1)

            if self.cuda:
                x = x.cuda()
                source = source.cuda()
                real_labels = real_labels.cuda()
                fake_labels = fake_labels.cuda()

            x = Variable(x)
            source = Variable(source)

            outputs = self.disc_model(x)

            generated = self.gen_model(source)
            gen_outputs = self.disc_model(generated.detach())

            gen_loss = self.loss_fn(gen_outputs,fake_labels)

            real_loss = self.loss_fn(outputs,real_labels)

            loss = gen_loss + real_loss
            loss.backward()
            self.disc_optimizer.step()

            self.disc_running_loss.add_(loss.cpu() * batch_size)
        

        #Extend __gen_train_func__
        def __gen_train_func__(self, data):

            super().__gen_train_func__(data)

            self.gen_optimizer.zero_grad()

            if isinstance(data, list) or isinstance(data, tuple):
                x = data[0]
            else:
                x = data
            batch_size = x.size(0)

            source = self.dist.sample((batch_size,self.latent_size))

            real_labels = torch.ones(batch_size,1)

            if self.cuda:
                source = source.cuda()
                real_labels = real_labels.cuda()

            source = Variable(source)

            fake_images = self.gen_model(source)
            outputs = self.disc_model(fake_images)

            loss = self.loss_fn(outputs,real_labels)
            loss.backward()

            self.gen_optimizer.step()

            self.gen_running_loss.add_(loss.cpu() * batch_size)

