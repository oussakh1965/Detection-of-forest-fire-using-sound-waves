import subprocess, os, time, signal

PATH = "/home/pi/projet-feux-de-foret/audio/"
FILE_PATH = os.path.join(PATH, 'record.wav')

def execute(record_time, sleeping_time):
    """Launch audio recording

    Args:
        record_time (int): number of seconds recoring
        sleeping_time (int): number of seconds sleeping after recording

    Returns:
        None
    """

    process_args = ['arecord', '-d', str(record_time), '-V' , 'mono' , '-v' , FILE_PATH]
    rec_process = subprocess.Popen(process_args, shell=False, preexec_fn=os.setsid)

    print("startRecordingArecord()> rec_proc pid= " + str(rec_process.pid))
    print("startRecordingArecord()> recording started.")

    time.sleep(record_time) #This sleeping time determines how long recording lasts
        
    os.killpg(rec_proc.pid, signal.SIGTERM)
    rec_proc.terminate()
    rec_proc = None
    print("stopRecordingArecord()> Recording stopped")

    time.sleep(sleeping_time)
    print("Done !")
    
    return None