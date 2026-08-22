#ifndef AppVersion
  #define AppVersion "dev"
#endif

#define ProjectRoot SourcePath + "..\"
#define ServerSource ProjectRoot + "dist\QwenASR-Server"
#define ArtifactBaseName "QwenASR-Server-" + AppVersion + "-win-x64"

[Setup]
AppId=QwenASR-Server
AppName=QwenASR Server
AppVersion={#AppVersion}
AppPublisher=QwenASR
DefaultDirName={localappdata}\Programs\QwenASR-Server
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
WizardStyle=modern
Uninstallable=no

OutputDir={#ProjectRoot}release
OutputBaseFilename={#ArtifactBaseName}
Compression=lzma2
SolidCompression=yes

DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "{#ServerSource}\*"; Excludes: "\config.json"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ProjectRoot}config.json.sample"; DestDir: "{app}"; DestName: "config.json"; Flags: ignoreversion
