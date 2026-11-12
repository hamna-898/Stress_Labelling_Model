@echo off
REM Full cloud labelling run: download -> audio models -> video model -> fuse -> send labels.
REM Each step uses the right environment automatically. Re-running skips finished work.
REM Optional:  run_cloud.bat --limit 2   (only process 2 new recordings)
cd /d "%~dp0"
set PIPELINE_SOURCE=cloud
set A=audio_env\Scripts\python.exe
set V=video_env\Scripts\python.exe

echo ===== 1/8 Download new recordings =====
%A% src\cloud_fetch.py %* || goto fail
echo ===== 2/8 Extract audio =====
%A% src\extract_audio.py || goto fail
echo ===== 3/8 Dpngtm (voice emotion) =====
%A% src\test_dpngtm.py || goto fail
echo ===== 4/8 audeering (voice arousal/valence) =====
%A% src\test_audeering.py || goto fail
echo ===== 5/8 eGeMAPS features =====
%A% src\test_egemaps.py || goto fail
echo ===== 6/8 py-feat (face) =====
%V% src\test_pyfeat.py || goto fail
echo ===== 7/8 Fuse votes =====
%A% src\fuse.py || goto fail
echo ===== 8/8 Send labels to backend =====
%A% src\cloud_send.py || goto fail
echo.
echo ALL STEPS DONE
goto end
:fail
echo.
echo Stopped at the step above. Fix the error and run again - finished work is skipped.
:end
pause
