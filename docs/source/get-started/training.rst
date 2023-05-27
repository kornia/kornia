.. _training_api:

Training API (experimental)
===========================

Kornia provides a Training API with the specific purpose to train and fine-tune the
supported deep learning algorithms within the library.

.. sidebar:: **Deep Alchemy**

    .. image:: https://github.com/kornia/data/raw/main/pixie_alchemist.png
       :width: 100%
       :align: center

  A seemingly magical process of transformation, creation, or combination of data to usable deep learning models.


.. important::
	In order to use our Training API you must: ``pip install kornia[x]``

Why a Training API ?
--------------------

Kornia includes deep learning models that eventually need to be updated through fine-tuning.
Our aim is to have an API flexible enough to be used across our vision models and enable us to
override methods or dynamically pass callbacks to ease the process of debugging and experimentations.

.. admonition:: **Disclaimer**
	:class: seealso

	We do not pretend to be a general purpose training library but instead we allow Kornia users to
	experiment with the training of our models.

Design Principles
-----------------

- `kornia` golden rule is to not have heavy dependencies.
- Our models are simple enough so that a light training API can fulfill our needs.
- Flexible and full control to the training/validation loops and customize the pipeline.
- Decouple the model definition from the training pipeline.
- Use plane PyTorch abstractions and recipes to write your own routines.
- Implement `accelerate <https://github.com/huggingface/accelerate/>`_ library to scale the problem.

Trainer Usage
-------------

The entry point to start traning with Kornia is through the :py:class:`~kornia.x.Trainer` class.

The main API is a self contained module that heavily relies on `accelerate <https://github.com/huggingface/accelerate/>`_
to easily scale the training over multi-GPUs/TPU/fp16 `(see more) <https://github.com/huggingface/accelerate#supported-integrations/>`_
by following standard PyTorch recipes. Our API expects to consume standard PyTorch components and you decide if `kornia` makes the magic
for you.

1. Define your model

.. code:: python

	model = nn.Sequential(
	  kornia.contrib.VisionTransformer(image_size=32, patch_size=16),
	  kornia.contrib.ClassificationHead(num_classes=10),
	)

2. Create the datasets and dataloaders for training and validation

.. code:: python

	# datasets
	train_dataset = torchvision.datasets.CIFAR10(
	  root=config.data_path, train=True, download=True, transform=T.ToTensor())

	valid_dataset = torchvision.datasets.CIFAR10(
	  root=config.data_path, train=False, download=True, transform=T.ToTensor())

	# dataloaders
	train_dataloader = torch.utils.data.DataLoader(
	  train_dataset, batch_size=config.batch_size, shuffle=True)

	valid_daloader = torch.utils.data.DataLoader(
	  valid_dataset, batch_size=config.batch_size, shuffle=True)

3. Create your loss function, optimizer and scheduler

.. code:: python

	# loss function
	criterion = nn.CrossEntropyLoss()

	# optimizer and scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
	  optimizer, config.num_epochs * len(train_dataloader)
	)

4. Create the Trainer and execute the training pipeline

.. code:: python

	trainer = kornia.train.Trainer(
	  model, train_dataloader, valid_daloader, criterion, optimizer, scheduler, config,
	)
	trainer.fit()  # execute your training !


Customize [callbacks]
---------------------

At this point you might think - *Is this API generic enough ?*

	Of course not ! What is next ? Let's have fun and **customize**.

The :py:class:`~kornia.x.Trainer` internals are clearly defined such in a way so that e.g you can
subclass and just override the :py:func:`~kornia.x.Trainer.evaluate` method and adjust
according to your needs. We provide predefined classes for generic problems such as
:py:class:`~kornia.x.ImageClassifierTrainer`, :py:class:`~kornia.x.SemanticSegmentationTrainer`.

.. note::
	More trainers will come as soon as we include more models.

You can easily customize by creating your own class, or even through ``callbacks`` as follows:

.. code:: python

    @torch.no_grad()
    def my_evaluate(self) -> dict:
      self.model.eval()
      for sample_id, sample in enumerate(self.valid_dataloader):
        source, target = sample  # this might change with new pytorch ataset structure

        # perform the preprocess and augmentations in batch
        img = self.preprocess(source)
        # Forward
        out = self.model(img)
        # Loss computation
        val_loss = self.criterion(out, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(out.detach(), target, topk=(1, 5))

    # create the trainer and pass the evaluate method as follows
    trainer = K.train.Trainer(..., callbacks={"evaluate", my_evaluate})

**Still not convinced ?**

	You can even override the whole :py:func:`~kornia.x.ImageClassifierTrainer.fit()`
	method and implement your custom for loops and the trainer will setup for you using the Accelerator all
	the data to the device and the rest of the story is just PyTorch :)

.. code:: python

    def my_fit(self, ):  # this is a custom pytorch training loop
      self.model.train()
      for epoch in range(self.num_epochs):
        for source, targets in self.train_dataloader:
          self.optimizer.zero_grad()

          output = self.model(source)
          loss = self.criterion(output, targets)

          self.backward(loss)
          self.optimizer.step()

          stats = self.evaluate()  # do whatever you want with validation

    # create the trainer and pass the evaluate method as follows
    trainer = K.train.Trainer(..., callbacks={"fit", my_fit})

.. note::
  The following hooks are available to override: ``preprocess``, ``augmentations``, ``evaluate``, ``fit``,
  ``on_checkpoint``, ``on_epoch_end``, ``on_before_model``


Preprocess and augmentations
----------------------------

Taking a pre-trained model from an external source and assume that fine-tuning with your
data by just changing few things in your model is usually a bad assumption in practice.

Fine-tuning a model need a lot tricks which usually means designing a good augmentation
or preprocess strategy before you execute the training pipeline. For this reason, we enable
through callbacks to pass pointers to the ``proprocess`` and ``augmentation`` functions to make easy
the debugging and experimentation experience.

.. code:: python

	def preprocess(x):
	  return x.float() / 255.

	augmentations = nn.Sequential(
	  K.augmentation.RandomHorizontalFlip(p=0.75),
	  K.augmentation.RandomVerticalFlip(p=0.75),
	  K.augmentation.RandomAffine(degrees=10.),
	  K.augmentation.PatchSequential(
		K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.8),
		grid_size=(2, 2),  # cifar-10 is 32x32 and vit is patch 16
		patchwise_apply=False,
	  ),
	)

	# create the trainer and pass the augmentation or preprocess
	trainer = K.train.ImageClassifierTrainer(...,
	  callbacks={"preprocess", preprocess, "augmentations": augmentations})

Callbacks utilities
-------------------

We also provide utilities to save checkpoints of the model or early stop the training. You can use
as follows passing as ``callbacks`` the classes :py:class:`~kornia.x.ModelCheckpoint` and
:py:class:`~kornia.x.EarlyStopping`.

.. code:: python

	model_checkpoint = ModelCheckpoint(
	  filepath="./outputs", monitor="top5",
	)

	early_stop = EarlyStopping(monitor="top5")

	trainer = K.train.ImageClassifierTrainer(...,
	  callbacks={"on_checkpoint", model_checkpoint, "on_epoch_end": early_stop})

Hyperparameter sweeps
---------------------

Use `hydra <https://hydra.cc>`_ to implement an easy search strategy for your hyper-parameters as follows:

.. note::

  Checkout the toy example in `here <https://github.com/kornia/kornia/tree/master/examples/train/image_classifier>`__

.. code:: python

  python ./train/image_classifier/main.py num_epochs=50 batch_size=32

.. code:: python

  python ./train/image_classifier/main.py --multirun lr=1e-3,1e-4

Distributed Training
--------------------

Kornia :py:class:`~kornia.x.Trainer` heavily relies on `accelerate <https://github.com/huggingface/accelerate/>`_ to
decouple the process of running your training scripts in a distributed environment.

.. note::

	We haven't tested yet all the possibilities for distributed training.
	Expect some adventures or `join us <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA>`_ and help to iterate :)

The below recipes are taken from the `accelerate` library in `here <https://github.com/huggingface/accelerate/tree/main/examples#simple-vision-example>`__:

- single CPU:

  * from a server without GPU

    .. code:: bash

      python ./train/image_classifier/main.py

  * from any server by passing `cpu=True` to the `Accelerator`.

    .. code:: bash

      python ./train/image_classifier/main.py --data_path path_to_data --cpu

  * from any server with Accelerate launcher

    .. code:: bash

      accelerate launch --cpu ./train/image_classifier/main.py --data_path path_to_data

- single GPU:

  .. code:: bash

    python ./train/image_classifier/main.py  # from a server with a GPU

- with fp16 (mixed-precision)

  * from any server by passing `fp16=True` to the `Accelerator`.

    .. code:: bash

      python ./train/image_classifier/main.py --data_path path_to_data --fp16

  * from any server with Accelerate launcher

    .. code:: bash

      accelerate launch --fp16 ./train/image_classifier/main.py --data_path path_to_data

- multi GPUs (using PyTorch distributed mode)

  * With Accelerate config and launcher

    .. code:: bash

      accelerate config  # This will create a config file on your server
      accelerate launch ./train/image_classifier/main.py --data_path path_to_data  # This will run the script on your server

  * With traditional PyTorch launcher

    .. code:: bash

      python -m torch.distributed.launch --nproc_per_node 2 --use_env ./train/image_classifier/main.py --data_path path_to_data

- multi GPUs, multi node (several machines, using PyTorch distributed mode)

  * With Accelerate config and launcher, on each machine:

    .. code:: bash

      accelerate config  # This will create a config file on each server
      accelerate launch ./train/image_classifier/main.py --data_path path_to_data  # This will run the script on each server

  * With PyTorch launcher only

    .. code:: bash

      python -m torch.distributed.launch --nproc_per_node 2 \
        --use_env \
        --node_rank 0 \
        --master_addr master_node_ip_address \
        ./train/image_classifier/main.py --data_path path_to_data  # On the first server

      python -m torch.distributed.launch --nproc_per_node 2 \
        --use_env \
        --node_rank 1 \
        --master_addr master_node_ip_address \
        ./train/image_classifier/main.py --data_path path_to_data  # On the second server

- (multi) TPUs

  * With Accelerate config and launcher

    .. code:: bash

      accelerate config  # This will create a config file on your TPU server
      accelerate launch ./train/image_classifier/main.py --data_path path_to_data  # This will run the script on each server

  * In PyTorch:
    Add an `xmp.spawn` line in your script as you usually do.
