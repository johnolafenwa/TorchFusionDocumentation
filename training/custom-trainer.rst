

Buiding Custom Trainers!
========================
While Tochfusion strives to provide very good trainers, we know researchers often need custom training logic.
TorchFusion makes using custom training logic easy. All you need to do is extend the Learners.


**Sample Custom Trainer** ::

 #Extend the StandardLearner
    class CustomLearner(StandardLearner):

        #Override the train logic
        def __train_func__(self, data):

            self.optimizer.zero_grad()

            if self.clip_grads is not None:
                clip_grads(self.model,self.clip_grads[0],self.clip_grads[1])

            train_x, train_y = data

            batch_size = train_x.size(0)

            train_x = Variable(train_x.cuda() if self.cuda else train_x)

            train_y = Variable(train_y.cuda() if self.cuda else train_y)

            outputs = self.model(train_x)
            loss = self.loss_fn(outputs, train_y)
            loss.backward()

            self.optimizer.step()

            self.train_running_loss.add_(loss.cpu() * batch_size)

            for metric in self.train_metrics:
                metric.update(outputs, train_y)
    
        #Override the evaluation logic
        def __eval_function__(self, data):

            test_x, test_y = data

            test_x = Variable(test_x.cuda() if self.cuda else test_x)

            test_y = Variable(test_y.cuda() if self.cuda else test_y)

            outputs = self.model(test_x)

            for metric in self.test_metrics:
                metric.update(outputs, test_y)
    
        #Override the validation logic

        def __val_function__(self, data):

            val_x, val_y = data
            val_x = Variable(val_x.cuda() if self.cuda else val_x)

            val_y = Variable(val_y.cuda() if self.cuda else val_y)

            outputs = self.model(val_x)

            for metric in self.val_metrics:
                metric.update(outputs, val_y)

        #override the prediction logic

        def __predict_func__(self, inputs):

            inputs = Variable(inputs.cuda() if self.cuda else inputs)

            return self.model(inputs)






