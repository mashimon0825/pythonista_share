# coding: utf-8

from objc_util import *
from ctypes import c_void_p
import ui
import time
#import numpy as np

# for CoreML
import requests
import os
import io
import photos
import dialogs
from PIL import Image
from objc_util import ObjCClass, nsurl, ns

MODEL_URL = 'https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3Tiny/YOLOv3Tiny.mlmodel'
MODEL_FILENAME = 'YOLOv3Tiny.mlmodel'
MODEL_PATH = os.path.join(os.path.expanduser('~/Documents'), MODEL_FILENAME)
MLModel = ObjCClass('MLModel')
VNCoreMLModel = ObjCClass('VNCoreMLModel')
VNCoreMLRequest = ObjCClass('VNCoreMLRequest')
VNImageRequestHandler = ObjCClass('VNImageRequestHandler')




# 全フレームを処理しようとすると動かなくなるのでこの程度で
FRAME_INTERVAL = 6  # 30fps / 6 = 5fps

frame_counter = 0
last_fps_time = time.time()
fps_counter = 0

AVCaptureSession = ObjCClass('AVCaptureSession')
AVCaptureDevice = ObjCClass('AVCaptureDevice')
AVCaptureDeviceInput = ObjCClass('AVCaptureDeviceInput')
AVCaptureVideoDataOutput = ObjCClass('AVCaptureVideoDataOutput')
AVCaptureVideoPreviewLayer = ObjCClass('AVCaptureVideoPreviewLayer')

CIImage    = ObjCClass('CIImage')
CIDetector = ObjCClass('CIDetector')

dispatch_get_current_queue = c.dispatch_get_current_queue
dispatch_get_current_queue.restype = c_void_p

CMSampleBufferGetImageBuffer = c.CMSampleBufferGetImageBuffer
CMSampleBufferGetImageBuffer.argtypes = [c_void_p]
CMSampleBufferGetImageBuffer.restype = c_void_p

CVPixelBufferLockBaseAddress = c.CVPixelBufferLockBaseAddress
CVPixelBufferLockBaseAddress.argtypes = [c_void_p, c_int]
CVPixelBufferLockBaseAddress.restype = None

CVPixelBufferGetWidth = c.CVPixelBufferGetWidth
CVPixelBufferGetWidth.argtypes = [c_void_p]
CVPixelBufferGetWidth.restype = c_int

CVPixelBufferGetHeight = c.CVPixelBufferGetHeight
CVPixelBufferGetHeight.argtypes = [c_void_p]
CVPixelBufferGetHeight.restype = c_int

CVPixelBufferUnlockBaseAddress = c.CVPixelBufferUnlockBaseAddress
CVPixelBufferUnlockBaseAddress.argtypes = [c_void_p, c_int]
CVPixelBufferUnlockBaseAddress.restype = None





def load_model():
        '''Helper method for downloading/caching the mlmodel file'''
        if not os.path.exists(MODEL_PATH):
                print(f'Downloading model: {MODEL_FILENAME}...')
                r = requests.get(MODEL_URL, stream=True)
                file_size = int(r.headers['content-length'])
                with open(MODEL_PATH, 'wb') as f:
                        bytes_written = 0
                        for chunk in r.iter_content(1024*100):
                                f.write(chunk)
                                print(f'{bytes_written/file_size*100:.2f}% downloaded')
                                bytes_written += len(chunk)
                print('Download finished')
        ml_model_url = nsurl(MODEL_PATH)
        c_model_url = MLModel.compileModelAtURL_error_(ml_model_url, None)
        ml_model = MLModel.modelWithContentsOfURL_error_(c_model_url, None)
        vn_model = VNCoreMLModel.modelForMLModel_error_(ml_model, None)
        return vn_model



def classify_img_data(img_data):
        '''The main image classification method, used by `classify_image` (for camera images) and `classify_asset` (for photo library assets).'''
        vn_model = load_model()
        # Create and perform the recognition request:
        req = VNCoreMLRequest.alloc().initWithModel_(vn_model).autorelease()
        #handler = VNImageRequestHandler.alloc().initWithData_options_(img_data, None).autorelease()
        handler = VNImageRequestHandler.alloc().initWithCIImage_options_(img_data, None).autorelease()

        success = handler.performRequests_error_([req], None)


        if success and len(req.results()) != 0  :

                results = []
                for num in range( len(req.results()) ) :

                    r_id       = str(req.results()[num].labels()[0].identifier())
                    r_conf     = req.results()[num].labels()[0].confidence()
                    r_ob_x     = req.results()[num].boundingBox().origin.x
                    r_ob_y     = req.results()[num].boundingBox().origin.y
                    r_ob_w     = req.results()[num].boundingBox().size.width
                    r_ob_h     = req.results()[num].boundingBox().size.height

                    results.append( [r_id, r_conf, r_ob_x, r_ob_y, r_ob_w, r_ob_h ] )

                return results

        else:
                return None



def captureOutput_didOutputSampleBuffer_fromConnection_(_self, _cmd, _output, _sample_buffer, _conn):
    global frame_counter, fps_counter, last_fps_time
    global image_width, image_height, faces

    # 性能確認のためビデオデータの実 FPS 表示
    fps_counter += 1
    now = time.time()
    if int(now) > int(last_fps_time):
        label_fps.text = '{:5.2f} fps'.format((fps_counter) / (now - last_fps_time))
        last_fps_time = now
        fps_counter = 0

    # 画像処理は FRAME_INTERVAL 間隔で処理
    if frame_counter == 0:
        # ビデオ画像のフレームデータを取得
        imagebuffer =  CMSampleBufferGetImageBuffer(_sample_buffer)
        # バッファをロック
        CVPixelBufferLockBaseAddress(imagebuffer, 0)

        image_width  = CVPixelBufferGetWidth(imagebuffer)
        image_height = CVPixelBufferGetHeight(imagebuffer)
        ciimage = CIImage.imageWithCVPixelBuffer_(ObjCInstance(imagebuffer))

        # CoreMLによる検出
        faces = classify_img_data(ciimage)

        # バッファのロックを解放
        CVPixelBufferUnlockBaseAddress(imagebuffer, 0)

        # 検出した顔の情報を使って表示を更新
        path_view.set_needs_display()

    frame_counter = (frame_counter + 1) % FRAME_INTERVAL

class PathView(ui.View):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FLAG = False
        self.ENTER = 0

    def draw(self):
        # 検出した顔の輪郭に合わせて、表示を加工
        #if faces is not None and faces.count() != 0:
        if faces is not None and len(faces) != 0:

#                # カメラの画像は X軸=1920 Y軸=1080
#                # View は X軸=375 Y軸=667
#                # 画像のX軸Y軸をViewのY軸X軸に対応させ、サイズを調整

            label_status.text = ''
            person_counter = 0

            for face in faces:
                label = face[0]
                confidence = round(face[1], 3)
                x = face[2]
                y = face[3]
                w = face[4]
                h = face[5]

                # CoreImageは座標系が左下０なので、左上０に変換する。
                y = 1 - y - h
                # boundingBox座標は正規化されているので戻す
                x = x * image_width
                y = y * image_height
                w = w * image_width
                h = h * image_height

                # 画像のX軸Y軸をViewのY軸X軸に対応させ、サイズを調整
                x2 = y    * self.width  / image_height
                y2 = x    * self.height / image_width
                w2 = h    * self.width / image_height
                h2 = w    * self.height  / image_width

                if label == 'person' :
                    COL='blue'
                    person_counter = person_counter + 1
                else :
                    COL='red'

                ui.set_color(COL)

                bb = ui.Path.rect(x2,y2,w2,h2)
                bb.stroke()
                ui.draw_string(label, rect=(x2,y2,w2,h2), color=COL )

                # enter count
                if label == 'person' :
                    if ( y2+h2/2 < self.height/2 - 60) :
                        self.FLAG = True
                    else :
                        if self.FLAG == True :
                            self.ENTER = self.ENTER + 1
                            self.FLAG = False


                    print( 'position:' +  str( y2+h2/2) )
                    print( 'border:' + str(self.height/2 - 60 ) )


        # border 
        ui.set_color('green')
        border = ui.Path.rect(0, self.height/2 - 60, self.width, 1)
        border.stroke()

                #print( 'position:' +  str( y2+h2/2) )
                #print( 'border:' + str(self.height/2 - 60 ) )

        label_status.text = str('person count(enter):' + str(self.ENTER) )





@on_main_thread
def main():
    global path_view, label_fps, faces, label_status

    # 画面の回転には対応しておらず
    # iPhoneの画面縦向きでロックした状態で、横長画面で使う想定
    # View のサイズは手持ちの iPhone6 に合わせたもの
    faces = None
    #main_view = ui.View(frame=(0, 0, 375, 667))
    main_view = ui.View(frame=(0, 0, 414, 896)) #iphone11
    path_view = PathView(frame=main_view.frame)
    main_view.name = 'CoreML by objc_util YOLO (trace)'

    sampleBufferDelegate = create_objc_class(
                                'sampleBufferDelegate',
                                methods=[captureOutput_didOutputSampleBuffer_fromConnection_],
                                protocols=['AVCaptureVideoDataOutputSampleBufferDelegate'])
    delegate = sampleBufferDelegate.new()

    session = AVCaptureSession.alloc().init()
    #device = AVCaptureDevice.defaultDeviceWithMediaType_('vide')

    inputDevices = AVCaptureDevice.devices()
    device = inputDevices[0] #背面カメラ
    #device = inputDevices[1] #前面カメラ


    print(device)


    _input = AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if _input:
        session.addInput_(_input)
    else:
        print('Failed to create input')
        return

    output = AVCaptureVideoDataOutput.alloc().init()
    queue = ObjCInstance(dispatch_get_current_queue())
    output.setSampleBufferDelegate_queue_(delegate, queue)
    output.alwaysDiscardsLateVideoFrames = True

    session.addOutput_(output)
    session.sessionPreset = 'AVCaptureSessionPresetHigh' # 1920 x 1080

    prev_layer = AVCaptureVideoPreviewLayer.layerWithSession_(session)
    prev_layer.frame = ObjCInstance(main_view).bounds()
    prev_layer.setVideoGravity_('AVLayerVideoGravityResizeAspectFill')

    ObjCInstance(main_view).layer().addSublayer_(prev_layer)

    # 性能確認のためビデオデータの実 FPS 表示
    label_fps = ui.Label(frame=(0, 0, main_view.width, 30), flex='W', name='fps')
    label_fps.background_color = (0, 0, 0, 0.5)
    label_fps.text_color = 'white'
    label_fps.text = ''
    label_fps.alignment = ui.ALIGN_CENTER

    # label_status : ステータス表示のための領域を追加
    label_status = ui.Label(frame=(0, 30, main_view.width, 30), flex='W', name='status')
    label_status.background_color = (0, 0, 0, 0.5)
    label_status.text_color = 'white'
    label_status.text = ''
    label_status.alignment = ui.ALIGN_CENTER


    main_view.add_subview(label_fps)
    main_view.add_subview(label_status)
    main_view.add_subview(path_view)

    session.startRunning()

    main_view.present('sheet')
    main_view.wait_modal()

    session.stopRunning()
    delegate.release()
    session.release()
    output.release()

if __name__ == '__main__':
    main()
