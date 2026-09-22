"""Fail before starting work if CUDA or a requested device cannot compute."""
import json
from lotaas_reprocessing.dedispersion import backend


def main():
    cp=backend('gpu')
    for index in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(index):
            x=cp.arange(1024,dtype=cp.float32)
            y=cp.fft.irfft(cp.fft.rfft(x),n=len(x))
            cp.testing.assert_allclose(x,y,atol=1e-3)
            props=cp.cuda.runtime.getDeviceProperties(index)
            print(json.dumps({'device':index,'name':props['name'].decode(),
                              'memory_bytes':props['totalGlobalMem'],'fft_verified':True}),flush=True)


if __name__=='__main__':main()
