```
git clone https://github.com/enesmsahin/diffusers.git
cd diffusers
pip install -r requirements.txt
pip install -e ".[torch]"
```
Do NOT delete diffusers folder. Python looks for files in that folder.

Run `inpaint_guided.py`. Modify `img_path`, `guidance_img_path`, `out_path` accordingly.