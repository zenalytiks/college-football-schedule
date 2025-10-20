module.exports = {
  apps: [{
    name: 'college-football-schedule',
    script: 'app.py',
    interpreter: 'python3',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      PYTHON_ENV: 'production',
      PORT: 8050
    },
    error_file: './logs/dash-error.log',
    out_file: './logs/dash-out.log',
    log_file: './logs/dash-combined.log',
    time: true,
    merge_logs: true,
    cron_restart: '0 3 * * *',
    restart_delay: 4000,
    min_uptime: '10s',
    max_restarts: 10,
    kill_timeout: 5000
  }]
};