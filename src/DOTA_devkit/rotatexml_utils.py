import os
import os.path as osp
import numpy as np
# from ..bbox import rbox2poly_single


def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):#size,gsd,imagesource#将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
    voc_headstr = """\
     <annotation>
        <folder>{}</folder>
        <filename>{}</filename>
        <path>{}</path>
        <source>
            <database>{}</database>
        </source>
        <size>
            <width>{}</width>
            <height>{}</height>
            <depth>{}</depth>
        </size>
        <segmented>0</segmented>
        """
    voc_rotate_objstr = """\
       <object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>{}</difficult>
		<robndbox>
			<cx>{}</cx>
			<cy>{}</cy>
			<w>{}</w>
			<h>{}</h>
			<angle>{}</angle>
		</robndbox>
		<extra>{:.2f}</extra>
	</object>
    """
    voc_tailstr = '''\
        </annotation>
        '''
    [floder,name]=os.path.split(img_name)
    # filename=name.replace('.jpg','.xml')
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
    head=voc_headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        obj=voc_rotate_objstr.format(CLASSES[int(box[6])],0,box[0],box[1],box[2],box[3],box[4],box[5])
        f.write(obj)
    f.write(voc_tailstr)
    f.close()
    

def result2rotatexml(results, dst_path, dataset):
    CLASSES = dataset.CLASSES
    img_names = dataset.img_names
    assert len(results) == len(
        img_names), 'length of results must equal with length of img_names'
    if not osp.exists(dst_path):
        os.mkdir(dst_path)
    
    for img_id, result in enumerate(results):
        rotateboxes=[]
        for class_id, bboxes in enumerate(result):
            if(bboxes.size != 0):
                for bbox in bboxes:
                    rotateboxes.append([bbox[0],bbox[1],bbox[3],bbox[2],bbox[4]+np.pi/2,bbox[5],class_id])
        rotateboxes=np.array(rotateboxes)
        # if rotateboxes.shape[0]>0:
            # inxw=rotateboxes[:,2]>5
            # inxh=rotateboxes[:,3]>12
            # inx=np.multiply(inxw,inxh)
            # rotateboxes=rotateboxes[inx]
        write_rotate_xml(dst_path,dataset.img_names[img_id],[1024 ,1024,3],0.5,'0.5',rotateboxes.reshape((-1,7)),CLASSES)
    return True
