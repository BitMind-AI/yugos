const path = require('node:path');

module.exports = {
  apps: [{
    name: 'yugos-validator',
    cwd: __dirname,
    script: path.join(__dirname, '.venv', 'bin', 'yugos-validator'),
    interpreter: 'none',
    args: [
      '--wallet', 'default',
      '--hotkey', 'default',
      '--version-key', '404020', // Set to the subnet's weights_version.
    ],
    instances: 1,
    watch: false,
    autorestart: true,
    restart_delay: 15000,
    kill_timeout: 180000,
  }],
};
