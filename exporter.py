import tensorflow as tf
import os
import numpy as np
import cv2

def fn_add_suffix(fn, suffix):
    folder = os.path.dirname(fn)
    base, ext = os.path.splitext(os.path.basename(fn))
    fn_new='%s_%s%s' %(base, suffix, ext)
    return os.path.join(folder, fn_new)


## Exports TF records for metric learning. Need pairs of matching images
## and labels
class TFRecordExporter:
    def __init__(self, high_res, low_res, fn_out):
        self.high_res=high_res
        high_res_log2=int(np.log2(high_res))
        low_res_log2=int(np.log2(low_res))
        opt = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
        self.writers=[]
        for r in range(high_res_log2, low_res_log2-1, -1):
            fn_out_cur=fn_add_suffix(fn_out, '%02d' % (r))
            print('Generating %s' %(fn_out_cur))
            self.writers.append(tf.io.TFRecordWriter(fn_out_cur, opt))
        self.cnt_im=0

    def close(self):
        for writer in self.writers:
            writer.close()
        self.writers = []
        print('%d images are added' %(self.cnt_im))

    def add_image(self, fn_im, match_im, label=None):
        try:
            im = cv2.imread(fn_im, cv2.IMREAD_COLOR)
            im=im[..., ::-1] #convert  BGR to RGB
            height, width, channel = im.shape
            print(self.cnt_im+1, fn_im, height, width, label)
            if height!=self.high_res or width!=self.high_res:
                im=cv2.resize(im, (self.high_res, self.high_res), interpolation = cv2.INTER_CUBIC if self.high_res > height else cv2.INTER_AREA)
            im = im.transpose(2, 0, 1) #HWC ->CHW
            im = im.astype(np.float32)

            match_im = cv2.imread(match_im, cv2.IMREAD_COLOR)
            match_im = match_im[..., ::-1] #convert  BGR to RGB
            height, width, channel = match_im.shape
            print(self.cnt_im + 1, match_im, height, width, label)
            if height != self.high_res or width != self.high_res:
                match_im = cv2.resize(match_im, (self.high_res, self.high_res), interpolation = cv2.INTER_CUBIC if self.high_res > height else cv2.INTER_AREA)
            match_im = match_im.transpose(2, 0, 1) #HWC ->CHW
            match_im = match_im.astype(np.float32)


            for i_writer, writer in enumerate(self.writers):
                if i_writer>0: # downsize
                    im = (im[:, 0::2, 0::2] + im[:, 0::2, 1::2] + im[:, 1::2, 0::2] + im[:, 1::2, 1::2]) * 0.25
                    match_im = (match_im[:, 0::2, 0::2] + match_im[:, 0::2, 1::2] + match_im[:, 1::2, 0::2] + match_im[:, 1::2, 1::2]) * 0.25

                quant = np.rint(im).clip(0, 255).astype(np.uint8)
                quant_match = np.rint(match_im).clip(0, 255).astype(np.uint8)
                feature={'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()])),
                        'match': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}

                # add label
                if label is not None:
                    feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                ex = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(ex.SerializeToString())
            self.cnt_im+=1
        except Exception as e:
            print(e)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == '__main__':
    high_res = 256
    low_res = 4
    add_label = True
    im_list_path = '/media/hermes/datashare/fl/face_vgg2/image_256/list_im.txt'
    id_list_path = '/media/hermes/datashare/fl/face_vgg2/image_256/list_identity.txt'
    id_names = [x.strip() for x in open(id_list_path).readlines()]
    identity_2_label = {x:i for i,x in enumerate(id_names)}

    list_im = [x.strip() for x in open(im_list_path).readlines()]
    list_ids = np.array([identity_2_label[x.split('/')[-2]] for x in list_im])

    fn_out_tf = '/media/hermes/dataspace/fl/dataset_tf/vgg2_metric/vgg2.tfrecords'

    with TFRecordExporter(high_res, low_res, fn_out_tf) as te:
        for idx, fn_im in enumerate(list_im):
            label = list_ids[idx] if add_label else None
            
            match_idxs = np.where(list_ids == label)[0]
            match_idxs = [mi for mi in match_idxs if mi != idx]
            try:
                if len(match_idxs) > 0:
                    sel_idx = np.random.choice(match_idxs)
                else:
                    sel_idx = int(idx)
            except:
                import pdb; pdb.set_trace()
            match_im = list_im[sel_idx]
            te.add_image(fn_im, match_im, label)
