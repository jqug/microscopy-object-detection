import caffe
import numpy as np
import glob

class ConvNetClassifier(caffe.Net):

    def __init__(self, opts):
                     
        model_file = '%s%s/deploy_data.prototxt' % (opts['model_dir'], opts['model'])
        pretrained_file = glob.glob('%s%s/*.caffemodel' % (opts['model_dir'], opts['model']))[0]
        mean = opts['mean']
        raw_scale = opts['raw_scale']
        input_scale = opts['input_scale']
        image_dims = opts['image_dims']
        channel_swap = opts['channel_swap']
        
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, inputs, oversample=True):
        
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = in_# caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            
            input_mirr_ = np.zeros((len(input_)*2, 50,50,1),dtype=np.float32)
                          
            for ix_, in_ in enumerate(input_):            
                input_mirr_[ix_*2] = in_
                input_mirr_[ix_*2+1] = np.fliplr(in_)
                #input_mirr_[ix_*4+1] = np.flipud(in_)
                #input_mirr_[ix_*4+2] = np.fliplr(in_)
                #input_mirr_[ix_*4+3] = np.fliplr(np.flipud(in_))
            input_ = input_mirr_
            
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]
            

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 2, 2, -1))
            predictions = predictions.mean(1)

        return predictions