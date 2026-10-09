param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$taskSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $taskVoice = $taskSynth.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1
    if (-not $taskVoice) { throw 'No English Windows speech voice is installed. Supply your own WAV recordings in data/downloads.' }
    $taskSynth.SelectVoice($taskVoice.VoiceInfo.Name)
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $taskTexts = @('The library will open on Monday morning.', 'We use machine learning to recognize speech and classify images.')
    for ($taskIndex = 0; $taskIndex -lt $taskTexts.Count; $taskIndex++) {
        $taskFile = Join-Path $OutputDirectory ('speech' + ($taskIndex + 1) + '.wav')
        $taskSynth.SetOutputToWaveFile($taskFile)
        $taskSynth.Speak($taskTexts[$taskIndex])
        $taskSynth.SetOutputToNull()
    }
    Write-Output ('Synthetic speech voice: ' + $taskVoice.VoiceInfo.Name)
} finally { $taskSynth.Dispose() }
