write-output "`nVARIAVEIS DO AMBIENTE DE DESENVOLVIMENTO`n"

# Configurações do Flask

$env:FLASK_APP = "main:create_api"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = 1

if (test-path env:FLASK_APP) {
    write-host "FLASK_APP: $env:FLASK_APP"
}
if (test-path env:FLASK_ENV) {
    write-host "FLASK_ENV: $env:FLASK_ENV"
}
if (test-path env:FLASK_DEBUG) {
    write-host "FLASK_DEBUG: $env:FLASK_DEBUG"
}



# Configurações do MongoDB
$env:MONGO_HOST_DEVELOP = "mongodb://localhost"
$env:MONGO_PORT_DEVELOP = "27017"
$env:MONGO_USER_DEVELOP = ""
$env:MONGO_PASSWORD_DEVELOP = ""
$env:MONGO_DATABASE_DEVELOP = "flame"

if (test-path env:MONGO_HOST_DEVELOP) {
    write-host "MONGO_HOST_DEVELOP: $env:MONGO_HOST_DEVELOP"
}
if (test-path env:MONGO_PORT_DEVELOP) {
    write-host "MONGO_PORT_DEVELOP: $env:MONGO_PORT_DEVELOP"
}
if (test-path env:MONGO_USER_DEVELOP) {
    write-host "MONGO_USER_DEVELOP: $env:MONGO_USER_DEVELOP"
}
if (test-path env:MONGO_PASSWORD_DEVELOP) {
    write-host "MONGO_PASSWORD_DEVELOP: $env:MONGO_PASSWORD_DEVELOP"
}
if (test-path env:MONGO_DATABASE_DEVELOP) {
    write-host "MONGO_DATABASE_DEVELOP: $env:MONGO_DATABASE_DEVELOP"
}



# Configurações do Storage
$env:STORAGE_HOST_DEVELOP = "sftp://harare.autoglass.com.br"
$env:STORAGE_PORT_DEVELOP = "22"
$env:STORAGE_USER_DEVELOP = "teste"
$env:STORAGE_PASSWORD_DEVELOP = "teste"
$env:STORAGE_LOCAL_PATH_DEVELOP = "files/images/frauds"
$env:STORAGE_REMOTE_PATH_DEVELOP = "bots/images/frauds"

if (test-path env:STORAGE_HOST_DEVELOP) {
    write-host "STORAGE_HOST_DEVELOP: $env:STORAGE_HOST_DEVELOP"
}
if (test-path env:STORAGE_PORT_DEVELOP) {
    write-host "STORAGE_PORT_DEVELOP: $env:STORAGE_PORT_DEVELOP"
}
if (test-path env:STORAGE_USER_DEVELOP) {
    write-host "STORAGE_USER_DEVELOP: $env:STORAGE_USER_DEVELOP"
}
if (test-path env:STORAGE_PASSWORD_DEVELOP) {
    write-host "STORAGE_PASSWORD_DEVELOP: $env:STORAGE_PASSWORD_DEVELOP"
}
if (test-path env:STORAGE_LOCAL_PATH_DEVELOP) {
    write-host "STORAGE_LOCAL_PATH_DEVELOP: $env:STORAGE_LOCAL_PATH_DEVELOP"
}
if (test-path env:STORAGE_REMOTE_PATH_DEVELOP) {
    write-host "STORAGE_REMOTE_PATH_DEVELOP: $env:STORAGE_REMOTE_PATH_DEVELOP"
}
