import whisper
import os

# model = whisper.load_model("base")
# result = model.transcribe("D:/robust_audio_ae/c_delta.wav")
# print("---" + result["text"] + "---")

path_file = 'E:/AdvExperiments/physical/honor_gohome/results2'   

# 遍历文件夹中的所有文件
for i in range(1, 51) :  
    success_count = 0
    fail_count = 0
    home = 0
    # 检查当前项是否为文件
    for file_name in os.listdir(path_file):
        if os.path.isfile(os.path.join(path_file, file_name)):
            # 检查文件扩展名是否为 WAV
            if file_name.endswith('.wav') and file_name.startswith(str(i)+'syn'):
                # 在这里对文件进行操作
                print(file_name)
                file_name_all = path_file + '\\' + file_name
                model = whisper.load_model("base")
                result = model.transcribe(file_name_all)
                print(result["text"])
                if (result["text"] == " Go home!" or result["text"] == " Go home." or result["text"] == " go home.") :
                    success_count += 1
                elif ("home" in result["text"]) :
                    home += 1
                else :
                    fail_count += 1
                print("success_count = ", success_count, " fail_count = ", fail_count, " home_count = ", home)  
    txt_path = path_file + '\\' + "record.txt"
    file = open(txt_path, "a")
    file.write("success = " + str(success_count) + "\thome = " + str(home) + "\tfail = " + str(fail_count) + '\n')
    file.close() 