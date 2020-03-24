# Modelos de Scripts de Automação

## Windows

'ATENÇÃO: Não suba as alterações realizadas neste diretório sem permissão, descarte as alterações antes de realizar o `commit`.'

* Os scripts foram testados somente no Windows 10 Professional.

Modifique os valores das variáveis de ambiente do script '''set_environment_variables.ps1''' de acordo com as configurações do servidores de homologação e produção.

A partir deste diretório, execute o comando `.\set_environment_variables_<ambiente>.ps1` para registrar as variáveis no sistema operacional.
* Exibe as variáveis registradas após a execução.
* Pode ser executado para atualizar ou remover as variáveis de ambiente.
* Altere o valor para "" (string vazia) para remover a variável de ambiente.

A partir deste diretório, execute o comando `.\get_environment_variables_<ambiente>.ps1` para exibir as variáveis que foram registradas no sistema operacional.
