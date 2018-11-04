from pydub import AudioSegment
from pydub.utils import make_chunks
import sys

if(len(sys.argv) == 4):
    dest = sys.argv[3]
    myAudio = AudioSegment.from_file(sys.argv[1], "wav")
    chunk_length_ms = int(sys.argv[2])*1000
    chunks = make_chunks(myAudio, chunk_length_ms)
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print("exporting " + chunk_name)
        chunk.export(dest+chunk_name, format="wav")

else:
    print("Takes three arguments: a wav file, the chunk length in seconds, and the destination directory")
