import sys, time
sys.path.insert(0, '.')
from runtime.executor import EdithTaskExecutor

executor = EdithTaskExecutor()
print('[*] Calling _spotify_play_track("Believer")...')
t0 = time.time()
res = executor._spotify_play_track("Believer")
print(f'[*] Result of _spotify_play_track: {res} in {time.time() - t0:.2f}s')
