write-output "`nVARIAVEIS DO AMBIENTE DE HOMOLOGACAO`n"

# Configurações do MongoDB
if (test-path env:MONGO_HOST_STAGING) {
    write-host "MONGO_HOST_STAGING: $env:MONGO_HOST_STAGING"
}
if (test-path env:MONGO_PORT_STAGING) {
    write-host "MONGO_PORT_STAGING: $env:MONGO_PORT_STAGING"
}
if (test-path env:MONGO_USER_STAGING) {
    write-host "MONGO_USER_STAGING: $env:MONGO_USER_STAGING"
}
if (test-path env:MONGO_PASSWORD_STAGING) {
    write-host "MONGO_PASSWORD_STAGING: $env:MONGO_PASSWORD_STAGING"
}
if (test-path env:MONGO_DATABASE_STAGING) {
    write-host "MONGO_DATABASE_STAGING: $env:MONGO_DATABASE_STAGING"
}

# Configurações do Storage
if (test-path env:STORAGE_HOST_STAGING) {
    write-host "STORAGE_HOST_STAGING: $env:STORAGE_HOST_STAGING"
}
if (test-path env:STORAGE_PORT_STAGING) {
    write-host "STORAGE_PORT_STAGING: $env:STORAGE_PORT_STAGING"
}
if (test-path env:STORAGE_USER_STAGING) {
    write-host "STORAGE_USER_STAGING: $env:STORAGE_USER_STAGING"
}
if (test-path env:STORAGE_PASSWORD_STAGING) {
    write-host "STORAGE_PASSWORD_STAGING: $env:STORAGE_PASSWORD_STAGING"
}