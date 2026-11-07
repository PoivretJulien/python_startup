<!DOCTYPE html>
<html lang="en">
<body>
<pre>
      
Quotes: 
      "IA accelerate things but should not forbid ourself to learn an share enthusiasm
in a more practical way to aboard the contemporary context with pragmatism,justness and 
numerical church community"



A real multi usage display that offer many possibility for a "all day feedback":
   
This is a reactive variant of the 3d system API,
simply one function decide what to draw in the frame only when an event trigger.
the app does not drain battery when idle.

support line, arrow, circle, point, point(with rgba value), images, text in (3d fields) and text on screen space.
(except for text) an anti-aliased version exist for each methods

Supported images format:
   PNG  JPEG/JPG/JPE/JFIF  BMP/DIB  GIF/APNG  WebP  TIFF/TIF  TGA  PSD
   ICO/CUR/ICNS  PCX  PPM/PGM/PBM/PNM  XBM/XPM  SGI/RGB/RGBA  RAS  QOI
   DDS  JP2/J2K/JPX/J2C/JPC  AVIF/AVIFS  FITS  HDF5  GRIB  EPS/PS/PDF
   WMF/EMF  MPO  PCD  PXR  FLC/FLI  BUFR  MSP  PALM

-> a reflection must be made at what is needed for learning and what are the alternatives 
for further technical investigation once the knowledge arrive.

but articulation is a king value at my sense and that's what python is designed for.
-> python has almost no compile time energy cost.
-> and the maths are program language agnostic anyway.

GPU backend (wgpu, no OpenGL):
   the app renders through wgpu-py, which drives the Rust wgpu-native library
   (wgpu.backends.wgpu_native, the module once called wgpu.backends.rs).
   The backend is picked automatically:
      macOS   -> Metal          (Apple's deprecated OpenGL is never touched)
      Linux   -> Vulkan         (wgpu's GL/GLES backend only if no Vulkan driver)
      Windows -> DX12, then Vulkan
   override with  --backend metal|vulkan|dx12|gl  or  WGPU_BACKEND=vulkan
   GLFW still owns the window + input (created with CLIENT_API = NO_API).
   The drawing API (draw_line, draw_points, draw_text, draw_image ...) is unchanged.
   The previous OpenGL version is kept in legacy_opengl/.

-> copy and paste the install script .sh or .ps1 + requirement.txt 

   macOS: ./setup-macos.sh      Linux: ./setup-linux.sh
   Windows (PowerShell): powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1
   check: ./.venv/bin/python check_env.py --window

run the install script, then launch the program with:
```
# (non reactive)
$ ./.venv/bin/python main.py --poll

# or reactive (by default)
$ ./.venv/bin/python main.py  
```
Edit the main.py at you convenience and play Vec2 and Vec3 in the function callback draw_frame(), 
just by tweaking the 1M particles system python show that it can achieve good 
enough performance for shading and linear algebra learning without unnecessary 
heavy machinery.  learning need workload strategy right ?
```python
def draw_frame() -> None:
      #custom code there.
   ...
```

</pre>
</body>
</html>

