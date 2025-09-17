# Configuration file for jupyter-kernel-gateway.

# c = get_config()  #noqa

# ------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
# ------------------------------------------------------------------------------
## This is an application.

## The date format used by logging formatters for %(asctime)s
#  Default: '%Y-%m-%d %H:%M:%S'
# c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  Default: '[%(name)s]%(highlevel)s %(message)s'
# c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
#  Default: 30
# c.Application.log_level = 30

## Configure additional log handlers.
#
#  The default stderr logs handler is configured by the log_level, log_datefmt
#  and log_format settings.
#
#  This configuration can be used to configure additional handlers (e.g. to
#  output the log to a file) or for finer control over the default handlers.
#
#  If provided this should be a logging configuration dictionary, for more
#  information see:
#  https://docs.python.org/3/library/logging.config.html#logging-config-
#  dictschema
#
#  This dictionary is merged with the base logging configuration which defines
#  the following:
#
#  * A logging formatter intended for interactive use called
#    ``console``.
#  * A logging handler that writes to stderr called
#    ``console`` which uses the formatter ``console``.
#  * A logger with the name of this application set to ``DEBUG``
#    level.
#
#  This example adds a new handler that writes to a file:
#
#  .. code-block:: python
#
#     c.Application.logging_config = {
#         "handlers": {
#             "file": {
#                 "class": "logging.FileHandler",
#                 "level": "DEBUG",
#                 "filename": "<path/to/file>",
#             }
#         },
#         "loggers": {
#             "<application-name>": {
#                 "level": "DEBUG",
#                 # NOTE: if you don't list the default "console"
#                 # handler here then it will be disabled
#                 "handlers": ["console", "file"],
#             },
#         },
#     }
#  Default: {}
# c.Application.logging_config = {}

## Instead of starting the Application, dump configuration to stdout
#  Default: False
# c.Application.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  Default: False
# c.Application.show_config_json = False

# ------------------------------------------------------------------------------
# JupyterApp(Application) configuration
# ------------------------------------------------------------------------------
## Base class for Jupyter applications

## Answer yes to any prompts.
#  Default: False
# c.JupyterApp.answer_yes = False

## Full path of a config file.
#  Default: ''
# c.JupyterApp.config_file = ''

## Specify a config file to load.
#  Default: ''
# c.JupyterApp.config_file_name = ''

## Generate default config file.
#  Default: False
# c.JupyterApp.generate_config = False

## The date format used by logging formatters for %(asctime)s
#  See also: Application.log_datefmt
# c.JupyterApp.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  See also: Application.log_format
# c.JupyterApp.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  See also: Application.log_level
# c.JupyterApp.log_level = 30

##
#  See also: Application.logging_config
# c.JupyterApp.logging_config = {}

## Instead of starting the Application, dump configuration to stdout
#  See also: Application.show_config
# c.JupyterApp.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  See also: Application.show_config_json
# c.JupyterApp.show_config_json = False

# ------------------------------------------------------------------------------
# KernelGatewayApp(JupyterApp) configuration
# ------------------------------------------------------------------------------
## Application that provisions Jupyter kernels and proxies HTTP/Websocket
#      traffic to the kernels.
#
#      - reads command line and environment variable settings
#      - initializes managers and routes
#      - creates a Tornado HTTP server
#      - starts the Tornado event loop

## Sets the Access-Control-Allow-Credentials header. (KG_ALLOW_CREDENTIALS env
#  var)
#  Default: ''
# c.KernelGatewayApp.allow_credentials = ''

## Sets the Access-Control-Allow-Headers header. (KG_ALLOW_HEADERS env var)
#  Default: ''
# c.KernelGatewayApp.allow_headers = ''

## Sets the Access-Control-Allow-Methods header. (KG_ALLOW_METHODS env var)
#  Default: ''
# c.KernelGatewayApp.allow_methods = ''

## Sets the Access-Control-Allow-Origin header. (KG_ALLOW_ORIGIN env var)
#  Default: ''
# c.KernelGatewayApp.allow_origin = ''

## Answer yes to any prompts.
#  See also: JupyterApp.answer_yes
# c.KernelGatewayApp.answer_yes = False

## Controls which API to expose, that of a Jupyter notebook server, the seed
#              notebook's, or one provided by another module, respectively using values
#              'kernel_gateway.jupyter_websocket', 'kernel_gateway.notebook_http', or
#              another fully qualified module name (KG_API env var)
#  Default: 'kernel_gateway.jupyter_websocket'
# c.KernelGatewayApp.api = 'kernel_gateway.jupyter_websocket'

## Authorization token required for all requests (KG_AUTH_TOKEN env var)
#  Default: ''
# c.KernelGatewayApp.auth_token = ''

## The authorizer class to use.
#  Default: 'jupyter_server.auth.authorizer.AllowAllAuthorizer'
# c.KernelGatewayApp.authorizer_class = 'jupyter_server.auth.authorizer.AllowAllAuthorizer'

## The base path for mounting all API resources (KG_BASE_URL env var)
#  Default: '/'
# c.KernelGatewayApp.base_url = '/'

## The full path to an SSL/TLS certificate file. (KG_CERTFILE env var)
#  Default: None
# c.KernelGatewayApp.certfile = None

## The full path to a certificate authority certificate for SSL/TLS client
#  authentication. (KG_CLIENT_CA env var)
#  Default: None
# c.KernelGatewayApp.client_ca = None

## Full path of a config file.
#  See also: JupyterApp.config_file
# c.KernelGatewayApp.config_file = ''

## Specify a config file to load.
#  See also: JupyterApp.config_file_name
# c.KernelGatewayApp.config_file_name = ''

## The random bytes used to secure cookies.
#          By default this is a new random number every time you start the server.
#          Set it to a value in a config file to enable logins to persist across server sessions.
#
#          Note: Cookie secrets should be kept private, do not share config files with
#          cookie_secret stored in plaintext (you can read the value from a file).
#  Default: b''
# c.KernelGatewayApp.cookie_secret = b''

## The file where the cookie secret is stored.
#  Default: ''
# c.KernelGatewayApp.cookie_secret_file = ''

## Default kernel name when spawning a kernel (KG_DEFAULT_KERNEL_NAME env var)
#  Default: ''
# c.KernelGatewayApp.default_kernel_name = ''

## Environment variables allowed to be inherited from the spawning process by the
#  kernel
#  Default: []
# c.KernelGatewayApp.env_process_whitelist = []

## Sets the Access-Control-Expose-Headers header. (KG_EXPOSE_HEADERS env var)
#  Default: ''
# c.KernelGatewayApp.expose_headers = ''

## Override any kernel name specified in a notebook or request
#  (KG_FORCE_KERNEL_NAME env var)
#  Default: ''
# c.KernelGatewayApp.force_kernel_name = ''

## Generate default config file.
#  See also: JupyterApp.generate_config
# c.KernelGatewayApp.generate_config = False

## The identity provider class to use.
#  Default: 'kernel_gateway.auth.identity.GatewayIdentityProvider'
# c.KernelGatewayApp.identity_provider_class = 'kernel_gateway.auth.identity.GatewayIdentityProvider'

## IP address on which to listen (KG_IP env var)
#  Default: '127.0.0.1'
# c.KernelGatewayApp.ip = '127.0.0.1'

## The kernel manager class to use.
#  Default: 'kernel_gateway.services.kernels.manager.SeedingMappingKernelManager'
# c.KernelGatewayApp.kernel_manager_class = 'kernel_gateway.services.kernels.manager.SeedingMappingKernelManager'

## The kernel spec manager class to use. Should be a subclass of
#  `jupyter_client.kernelspec.KernelSpecManager`.
#  Default: 'jupyter_client.kernelspec.KernelSpecManager'
# c.KernelGatewayApp.kernel_spec_manager_class = 'jupyter_client.kernelspec.KernelSpecManager'

## The kernel websocket connection class to use.
#  Default: 'jupyter_server.services.kernels.connection.channels.ZMQChannelsWebsocketConnection'
# c.KernelGatewayApp.kernel_websocket_connection_class = 'jupyter_server.services.kernels.connection.channels.ZMQChannelsWebsocketConnection'

## The full path to a private key file for usage with SSL/TLS. (KG_KEYFILE env
#  var)
#  Default: None
# c.KernelGatewayApp.keyfile = None

## The date format used by logging formatters for %(asctime)s
#  See also: Application.log_datefmt
# c.KernelGatewayApp.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#  See also: Application.log_format
# c.KernelGatewayApp.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#  See also: Application.log_level
# c.KernelGatewayApp.log_level = 30

##
#  See also: Application.logging_config
# c.KernelGatewayApp.logging_config = {}

## Sets the Access-Control-Max-Age header. (KG_MAX_AGE env var)
#  Default: ''
# c.KernelGatewayApp.max_age = ''

## Limits the number of kernel instances allowed to run by this gateway.
#  Unbounded by default. (KG_MAX_KERNELS env var)
#  Default: None
# c.KernelGatewayApp.max_kernels = None

## Port on which to listen (KG_PORT env var)
#  Default: 8888
# c.KernelGatewayApp.port = 8888

## Number of ports to try if the specified port is not available (KG_PORT_RETRIES
#  env var)
#  Default: 50
# c.KernelGatewayApp.port_retries = 50

## Number of kernels to prespawn using the default language. No prespawn by
#  default. (KG_PRESPAWN_COUNT env var)
#  Default: None
# c.KernelGatewayApp.prespawn_count = None

## Runs the notebook (.ipynb) at the given URI on every kernel launched. No seed
#  by default. (KG_SEED_URI env var)
#  Default: None
# c.KernelGatewayApp.seed_uri = None

## Instead of starting the Application, dump configuration to stdout
#  See also: Application.show_config
# c.KernelGatewayApp.show_config = False

## Instead of starting the Application, dump configuration to stdout (as JSON)
#  See also: Application.show_config_json
# c.KernelGatewayApp.show_config_json = False

## Sets the SSL version to use for the web socket connection. (KG_SSL_VERSION env
#  var)
#  Default: None
# c.KernelGatewayApp.ssl_version = None

## Use x-* header values for overriding the remote-ip, useful when application is
#  behind a proxy. (KG_TRUST_XHEADERS env var)
#  Default: False
# c.KernelGatewayApp.trust_xheaders = False

## Specifies the ping interval(in seconds) that should be used by zmq port
#                                       associated with spawned kernels. Set this variable to 0 to disable ping mechanism.
#                                      (KG_WS_PING_INTERVAL_SECS env var)
#  Default: 30
# c.KernelGatewayApp.ws_ping_interval = 30

# ------------------------------------------------------------------------------
# NotebookHTTPPersonality(LoggingConfigurable) configuration
# ------------------------------------------------------------------------------
## Personality for notebook-http support, creating REST endpoints
#      based on the notebook's annotated cells

## Optional API to download the notebook source code in notebook-http mode,
#  defaults to not allow
#  Default: False
# c.NotebookHTTPPersonality.allow_notebook_download = False

## Determines which module is used to parse the notebook for endpoints and
#              documentation. Valid module names include 'kernel_gateway.notebook_http.cell.parser'
#              and 'kernel_gateway.notebook_http.swagger.parser'. (KG_CELL_PARSER env var)
#  Default: 'kernel_gateway.notebook_http.cell.parser'
# c.NotebookHTTPPersonality.cell_parser = 'kernel_gateway.notebook_http.cell.parser'

## Maps kernel language to code comment syntax
#  Default: {'scala': '//', None: '#'}
# c.NotebookHTTPPersonality.comment_prefix = {'scala': '//', None: '#'}

## Serve static files on disk in the given path as /public, defaults to not serve
#  Default: None
# c.NotebookHTTPPersonality.static_path = None

# ------------------------------------------------------------------------------
# JupyterWebsocketPersonality(LoggingConfigurable) configuration
# ------------------------------------------------------------------------------
## Personality for standard websocket functionality, registering
#      endpoints that are part of the Jupyter Kernel Gateway API

## Environment variables allowed to be set when
#                               a client requests a new kernel
#  Default: []
# c.JupyterWebsocketPersonality.env_whitelist = []

## Permits listing of the running kernels using API endpoints /api/kernels
#              and /api/sessions (KG_LIST_KERNELS env var). Note: Jupyter Notebook
#              allows this by default but kernel gateway does not.
#  Default: False
# c.JupyterWebsocketPersonality.list_kernels = False
