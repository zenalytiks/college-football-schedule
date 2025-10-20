module.exports = {
  apps: [{
    name: 'college-football-schedule',
    script: 'app.py',
    interpreter: './venv/bin/python',
    cwd: './',
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      PYTHONUNBUFFERED: '1',
      PORT: '8050'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    cron_restart: '0 3 * * *',
    restart_delay: 4000
  }]
};