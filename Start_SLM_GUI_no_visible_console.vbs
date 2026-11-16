' Launches the SLM GUI with no visible console window.
' Python is started directly (not through Start_SLM_GUI.bat) so no cmd.exe stays
' alive as its parent: Task Manager then lists the GUI as its own app, with its icon.
' The GUI's output is shown in its Output dock; use Start_SLM_GUI.bat to also see a console.
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonPath = fso.BuildPath(scriptDir, "SLM_venv\Scripts\python.exe")
guiPath = fso.BuildPath(scriptDir, "GUI_SLM.py")
If Not fso.FileExists(pythonPath) Then
    MsgBox "Virtual environment not found:" & vbCrLf & pythonPath, vbCritical, "SLM Controller"
    WScript.Quit 1
End If
shell.CurrentDirectory = scriptDir
shell.Run """" & pythonPath & """ """ & guiPath & """", 0, False
