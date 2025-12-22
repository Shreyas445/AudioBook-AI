from pydub import AudioSegment
import os
from typing import List, Dict

class Studio:
    def __init__(self):
        pass

    def mix(self, 
            voice_files: List[str], 
            music_files: List[str], 
            output_file: str):
        """
        Stitches voice segments and overlays background music with ducking.
        """
        
        voice_track = AudioSegment.empty()
        music_track = AudioSegment.empty()
        
        # Pause between sentences (in ms) coverage by BGM
        PAUSE_MS = 250 

        for i, v_path in enumerate(voice_files):
            if not os.path.exists(v_path):
                print(f"Missing voice file: {v_path}, skipping.")
                continue
                
            # 1. Prepare Voice Segment (with pause)
            v_seg = AudioSegment.from_wav(v_path)
            # Add pause to the voice track so speech isn't breathless
            v_block = v_seg + AudioSegment.silent(duration=PAUSE_MS)
            
            block_duration = len(v_block)
            voice_track += v_block
            
            # 2. Prepare Music Segment (matched to Voice block duration)
            m_path = music_files[i] if i < len(music_files) else None
            
            if m_path and os.path.exists(m_path):
                m_source = AudioSegment.from_wav(m_path)
                
                # Loop music to cover the entire voice block (including the pause)
                # This ensures music plays continuously THROUGH the silence
                if len(m_source) < block_duration:
                    loops = (block_duration // len(m_source)) + 2
                    m_long = m_source * loops
                    m_chunk = m_long[:block_duration]
                else:
                    m_chunk = m_source[:block_duration]
                
                # Apply a micro-fade to avoid clicking at segment boundaries
                m_chunk = m_chunk.fade_in(10).fade_out(10)
                
                music_track += m_chunk
            else:
                # No music? Add silence to keep sync with voice track
                music_track += AudioSegment.silent(duration=block_duration)
        
        # 3. Final Mix
        # Ducking: Reduce music volume
        final_audio = voice_track.overlay(music_track - 18)
        
        # Export
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"Exporting full audiobook to {output_file}...")
        final_audio.export(output_file, format="mp3")
        return output_file

if __name__ == "__main__":
    # Test stub
    # studio = Studio()
    # studio.mix(["v1.wav"], ["m1.wav"], "out.mp3")
    pass
