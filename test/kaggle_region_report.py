"""Paste after region experiments to collect measurements; cache tensors stay on Kaggle."""
from pathlib import Path
from datetime import datetime
import zipfile
from IPython.display import FileLink, Image, display
project=Path('/kaggle/working/KD-SBIR-AVKD')
directories=sorted((project/'tb_logs').glob('region_*/version_*/region_diagnostics'))
if not directories:
    raise FileNotFoundError('No region diagnostics under current project tb_logs')
archive=Path('/kaggle/working')/('region_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for directory in directories:
        print('Including:',directory)
        for path in sorted(directory.iterdir()):
            if path.is_file():bundle.write(path,path.relative_to(project))
        curve=directory/'training_diagnostics.png'
        if curve.is_file():display(Image(filename=str(curve)))
    for path in sorted((project/'saved_models').glob('region_*/run.json')):
        bundle.write(path,path.relative_to(project))
    for directory in sorted(Path('/kaggle/working/teacher_cache').glob('*semantic_regions*')):
        for name in ('manifest.json','teacher_evaluation.json'):
            path=directory/name
            if path.is_file():bundle.write(path,'cache/'+directory.name+'/'+name)
print('Measurements and figures:',archive)
display(FileLink(str(archive)))
