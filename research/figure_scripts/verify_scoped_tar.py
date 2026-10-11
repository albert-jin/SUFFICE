import argparse,hashlib,json,tarfile
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args()
    audit=json.loads(a.audit.read_text());assert a.archive.stat().st_size==audit['bytes'] and sha(a.archive)==audit['sha256']
    a.outdir.mkdir(parents=True,exist_ok=True);dest=a.outdir.resolve()
    with tarfile.open(a.archive) as tf:
        for m in tf.getmembers():assert m.isfile() and (dest/m.name).resolve().is_relative_to(dest)
        tf.extractall(dest,filter='data')
    manifest=json.loads((dest/a.manifest).read_text())
    for row in manifest['files']:
        p=dest/row['path'];assert p.resolve().is_relative_to(dest) and sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],row['path']
    result={'archive_sha256':audit['sha256'],'bytes':audit['bytes'],'verified_files':len(manifest['files']),'outdir':str(dest)}
    (dest/'LOCAL_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
