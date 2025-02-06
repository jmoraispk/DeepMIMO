'''output_dir = os.path.abspath(os.path.join(self.output_folder, folder_tag + '_' + str(i)))
command_start = '"' + self.WI_exe + '" -f "' + self.xml_path + '" -out '
command_end = ' -p ' + self.name
command = command_start + '"' + output_dir + '"' + command_end
if generate_data:
    subprocess.call(command, shell=True)
# print(command)
# "C:\\Program Files\\Remcom\\Wireless InSite 3.3.0.4\\bin\\calc\\wibatch.exe" -f "C:\\Users\\sjiang74\\GitHub\\DeepMIMO_large_tmp\\scenario\\deepsense-synth-proj.study_area.xml" -out "C:\\Users\\sjiang74\\GitHub\\DeepMIMO_large_tmp\\scenario\\output_data\\run_100" -p deepsense-synth-proj
return output_dir'''
