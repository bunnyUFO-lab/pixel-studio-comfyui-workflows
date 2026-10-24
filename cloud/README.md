# ComfyUI on Modal (cloud GPU)

`modal_comfy.py` runs the same ComfyUI as the Docker setup on a [Modal](https://modal.com) cloud GPU (L40S, 48 GB),
for when you have no local GPU or it is busy. Models are downloaded inside Modal, and the server scales to zero when
idle, so you pay only while it runs.

## Setup

1. Create a Modal account, add a payment method and set a workspace budget as a spending limit.
2. Install the Modal client and sign in:

   ```bash
   pip install modal
   modal setup
   ```

3. Download the models (`--workflows pixel` by default; `all` or a comma-separated list also work):

   ```bash
   modal run cloud/modal_comfy.py::download
   ```

4. Deploy; it prints the server's URL:

   ```bash
   modal deploy cloud/modal_comfy.py
   ```

5. In the Modal dashboard, create a **Proxy Auth Token**. Requests must send it as the `Modal-Key` and
   `Modal-Secret` headers.

## Use

Connect to the URL like any ComfyUI server, with the two headers. The first request after a quiet spell wakes the
GPU (about half a minute), and it shuts down again 5 minutes after the last request.

To take it down: `modal app stop sprite-studio-comfy`.
