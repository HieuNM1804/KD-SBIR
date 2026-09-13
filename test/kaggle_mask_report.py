"""Paste in a Kaggle cell after experiments to download measurements and figures."""
from datetime import datetime
from pathlib import Path
import zipfile
from IPython.display import FileLink, Image, display

project=Path('/kaggle/working/KD-SBIR-AVKD')
directories=sorted((project/'tb_logs').glob('mask_*/version_*/mask_diagnostics'))
if not directories:
    raise FileNotFoundError('No mask diagnostics found under the current project tb_logs')
archive=Path('/kaggle/working')/('mask_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for directory in directories:
        print('Including:',directory)
        for path in sorted(directory.iterdir()):
            if path.is_file():bundle.write(path,path.relative_to(project/'tb_logs'))
        curve=directory/'training_diagnostics.png'
        if curve.is_file():display(Image(filename=str(curve)))
print('Diagnostics only; checkpoints remain in saved_models:',archive)
display(FileLink(str(archive)))
