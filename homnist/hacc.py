from __future__ import print_function
import serial
import time
import numpy as np

ser = None

RESET = 1
WRITEDATA = 2
WLSET = 3
CONFIGUREROW = 4
CONFIGURECOL = 5
RUN = 6
ADCREAD = 7
DISABLEADC = 8
CONFIGUREACC = 9
CONFIGURECAP = 10
DISABLEDTC = 11
CONFIGUREIOCLK = 13
CONFIGUREDTCCLK = 14
RUN_IMAGE = 15
WRITE_IMAGE = 16
READ_OUT_IMAGE = 17
READ_STORED_DATA = 18
RUN_IMAGE_NN = 19
READ_OUT_IMAGE_NN = 20
CP_OUT_IN = 21
WRITE_CNN = 22
RUN_CNN = 23
READ_OUT_IMAGE_CNN = 24
WRITE_IMAGE_CNN = 25
TESTING = 50

def init_serial(port='/dev/ttyUSB2', baudrate=1500000):
    global ser
    
    if ser is None:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate  # 1500000  # 921600  # 115200,
        )

def send_int(number):
    ser.write(number.to_bytes(4, byteorder="big"))
    
def send_byte(number):
    ser.write(int(number).to_bytes(1, byteorder="big"))

def send_int_reverse(number):
    ser.write(number.to_bytes(4, byteorder="little"))
    

def read_byte():  # Reads one byte and converts it to int
    number = int.from_bytes(ser.read(1), "little")
    return number


def reset():
    send_byte(RESET)
    ack = read_byte()
    return ack


def configureioclk(factor):
    send_byte(CONFIGUREIOCLK)
    read_byte()
    send_int(factor)
    ack = read_byte()
    return ack

def configuredtcclk(factor):
    send_byte(CONFIGUREDTCCLK)
    read_byte()
    send_int(factor)
    ack = read_byte()
    return ack

def confaccrelu(accreluconfiguration):
    send_byte(CONFIGUREACC)
    send_int(accreluconfiguration)
    ack = read_byte()
    return ack


def send_CNN(weight_files_list):
    nlayers = len(weight_files_list)
    
    send_byte(WRITE_CNN)
    send_byte(nlayers)
    rcv_nlayers = read_byte()
    
    if(rcv_nlayers != nlayers):
        raise ValueError('Error in synchronizing byte send_CNN nlayers. Expected {}, received number = {}'.format(nlayers, rcv_nlayers))
    
    else:
        # print("Start synchronization byte ok. Number of received layers = {}".format(rcv_nlayers))
        for layer_num, layer_file in enumerate(weight_files_list):
            rcv_layer = read_byte()
            assert layer_num == rcv_layer, 'Error in synchronizing byte send_CNN rcv_layer. Expected {}, received number = {}'.format(layer_num, rcv_layer)
            
            # print("LAYER = {}; Received layer = {}".format(LAYER, rcv_layer))
            layer_dict = read_weightfile(layer_file)
            confcap_int = int(layer_dict['cap'][::-1], 2)
            W_converted = convertW(layer_dict['weights_tensor'])
            
            send_byte(layer_dict['nrows'])
            send_byte(layer_dict['ncols'])
            send_byte(layer_dict['initial_rc'])
            send_byte(layer_dict['stride'])
            send_int(confcap_int)
            
            for col in range(layer_dict['ncols']):
                for element in range(23):
                    send_int(W_converted[col][element])
                    
        rcv = read_byte()
        if (rcv != 79):
            raise ValueError('Synchronization error end CNN write')
        
        else:
            print("CNN write to the FPGA ok")
        
        return

def run_cnn():
    send_byte(RUN_CNN)
    return


def image_to_compu(image, destination_size=(16, 16, 16)):
    compu_img = np.rint(image).astype(int)
    
    # We pad with zeros if necessary
    input_size = compu_img.shape
    required_padding = np.array(destination_size) - np.array(input_size)
    if required_padding.sum() > 0:
        pad_vals = ((0, required_padding[0]), (0, required_padding[1]), (0, required_padding[2]))
        compu_img = np.pad(compu_img, pad_vals, mode='constant')
    
    compu_img = np.transpose(compu_img, [2, 0, 1])  # From [height, width, channels] to [channels, height, width]
    
    return compu_img


def send_image_cnn(image):  # Sends an image of 16x16x16. The image must be a matrix in where each row is a channel and each column is an array of 1024 pixel values
    nch = image.shape[2]
    compu_img = image_to_compu(image)
    
    _, nrow, ncol = compu_img.shape
    
    send_byte(WRITE_IMAGE_CNN)
    send_byte(nch)
    
    for ch in range(nch):
        for row in range(nrow):
            for col in range(ncol):
                send_byte(compu_img[ch][row][col])
    
    return


def read_output_image_cnn():
    send_byte(READ_OUT_IMAGE_CNN)
    nch = read_byte()
    imsize = read_byte()
    
    out_image = np.zeros((nch, imsize, imsize))
    for ch in range(nch):
        for row in range(imsize):
            for col in range(imsize):
                out_image[ch][row][col] = read_byte()
    
    return out_image



def read_weightfile(filename):  # Reads the weights file
    with open(filename) as f:
        lines = [line.strip() for line in f]
    
    weights = []
    for l in lines:
        if l.startswith('//'):
            pass  # We ignore the comment
        
        elif l.startswith('#RC'):
            l = l.split(' ')
            nrows = int(l[1].strip())
            ncols = int(l[2].strip())
        
        elif l.startswith('#PS'):
            l = l.split(' ')
            initial_rc = int(l[1].strip())
            stride = int(l[2].strip())
        
        elif l.startswith('#CAP'):
            l = l.split(' ')
            cap = l[1].strip()
        
        else:
            kernel_weights = l.split(' ')
            weights.append(kernel_weights)
    
    assert len(weights) == (ncols * nrows)
    kernel_size = len(weights[0])
    
    weights_tensor = np.zeros([ncols, nrows, kernel_size])
    
    for c in range(ncols):
        for r in range(nrows):
            for k in range(kernel_size):
                weights_tensor[c, r, k] = weights[c * nrows + r][kernel_size - k - 1]
    
    out_dict = {'weights_tensor': weights_tensor,
                'ncols': ncols,
                'nrows': nrows,
                'initial_rc': initial_rc,
                'stride': stride,
                'cap': cap}
    
    return out_dict


def convertW(W):  # Converts the read matrix of weights to 23 words of 32 bit to be send to the FPGA
    #First of all, convert all negative numbers into positive with sign
    ncols = W.shape[0]
    nrows = W.shape[1]
    nmult = W.shape[2]
    
    #print("ncols = {}, nrows = {} and nmulti = {}".format(ncols, nrows, nmult))
    
    #DATAIN_CHIP=np.zeros((ncols,nrows,nmult))#Data to be sent to the chip
    DATAIN_CHIP=[]
    
    for COL in range(ncols):
        for KP in range(nrows):
            for MULT in range(nmult):
                if W[COL][KP][MULT]<0:
                    #If a weight is negative, it is written as 16-weight (negative values are in the range 16-31)
                    W[COL][KP][MULT]=16-W[COL][KP][MULT]
                    #print("W[{}][{}][{}] = {}".format(COL, KP, MULT, W[COL][KP][MULT]))
   
    for COL in range(ncols):
        a=""
        datain=[]
        N=0
        for KP in range (nrows):
            for MULT in range (nmult):
                a=a+f'{int(W[COL][KP][MULT]):05b}'[::-1]#We add the five bits but reversed
                N=N+1
        for i in range ((16-nrows)*9):
            a=a+f'{0:05b}'
        #Adding the additional bits equal to zero to fill the 23 input registers in the FPGA (23*32=736 bits)
        a=a+"0000000000000000"
        a=[*a] #Split the bits
        temp=""
    
        for i in range(23):  # Divide the input data in blocks of 32 bits
            temp=""
            for k in range(32):
                temp=temp+a[735-32*i-k]
            #print("Bits to be sent to the FPGA:")
            #print(temp)
            datain.append(int(temp,2))#Convert the 32 bit words into integers
        DATAIN_CHIP.append(datain[::-1])
    return DATAIN_CHIP
