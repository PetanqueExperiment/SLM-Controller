' Launches Start_SLM_GUI.bat with no visible console window
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(scriptDir, "Start_SLM_GUI.bat")
shell.Run """" & batPath & """", 0, False
