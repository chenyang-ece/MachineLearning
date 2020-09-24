from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=200,height=200):
    img=Image.open(jpgfile)

    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))

for jpgfile in glob.glob("brick picture/others/*.jpg"):
    convertjpg(jpgfile,"4")