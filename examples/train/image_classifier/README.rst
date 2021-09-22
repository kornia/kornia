Image Classifier Example
========================

This is a toy example implementing an image classification application.

1. Install dependencies

```bash
pip install -r requirements.tx
```

2. Execute the script: The entry point to this example is the file

```bash
python main.py
```

3. Modify the hyper-parameters in `config.yml` and execute

```bash
python main.py num_epochs=50
```

4. Sweep hyper-parameters

```bash
python main.py --multirun num_epochs=1 lr=1e-3,1e-4
```

Explore hydra to make cool stuff with the config files: https://hydra.cc/
