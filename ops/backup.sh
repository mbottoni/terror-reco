#!/bin/sh
# Nightly Postgres backup, run by the `backup` service in docker-compose.prod.yml.
#
#   backup.sh loop      take a dump every day at BACKUP_HOUR_UTC (the service's command)
#   backup.sh once      take one dump now
#   backup.sh verify    restore the newest dump into a scratch database and
#                       compare row counts with the live one, then drop it
#
# Dumps land in /backups (a host directory), custom format, 14 kept.
# That protects against a bad migration, a mistaken DELETE or a corrupted
# table. It does NOT protect against losing the server: for that, set the
# four BACKUP_S3_* variables in .env.prod and every dump is also copied to
# S3-compatible storage (Cloudflare R2, Backblaze B2, AWS S3).
set -eu

: "${PGHOST:=db}" "${PGUSER:=terror}" "${PGDATABASE:=terror}"
: "${BACKUP_DIR:=/backups}" "${BACKUP_KEEP:=14}" "${BACKUP_HOUR_UTC:=3}"
export PGHOST PGUSER PGDATABASE

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) backup: $*"; }

upload() {
  # Off-box copy, only when configured. rclone is installed on first use so
  # an unconfigured deployment pulls nothing extra.
  [ -n "${BACKUP_S3_BUCKET:-}" ] || { log "no BACKUP_S3_BUCKET set: dump stays on this server only"; return 0; }
  command -v rclone >/dev/null 2>&1 || apk add --no-cache rclone >/dev/null
  export RCLONE_CONFIG_OFFBOX_TYPE=s3
  export RCLONE_CONFIG_OFFBOX_PROVIDER="${BACKUP_S3_PROVIDER:-Cloudflare}"
  export RCLONE_CONFIG_OFFBOX_ENDPOINT="${BACKUP_S3_ENDPOINT:?set BACKUP_S3_ENDPOINT}"
  export RCLONE_CONFIG_OFFBOX_ACCESS_KEY_ID="${BACKUP_S3_ACCESS_KEY_ID:?set BACKUP_S3_ACCESS_KEY_ID}"
  export RCLONE_CONFIG_OFFBOX_SECRET_ACCESS_KEY="${BACKUP_S3_SECRET_ACCESS_KEY:?set BACKUP_S3_SECRET_ACCESS_KEY}"
  rclone copy "$1" "offbox:${BACKUP_S3_BUCKET}/terror-reco/" --quiet
  # Off-box retention is longer than on-box: 60 days.
  rclone delete "offbox:${BACKUP_S3_BUCKET}/terror-reco/" --min-age 60d --quiet || true
  log "uploaded $(basename "$1") to ${BACKUP_S3_BUCKET}"
}

once() {
  mkdir -p "$BACKUP_DIR"
  stamp=$(date -u +%Y%m%d-%H%M%S)
  target="$BACKUP_DIR/terror-$stamp.dump"
  # Write to a temp name first: a dump interrupted halfway must never look
  # like a finished backup.
  pg_dump --format=custom --no-owner --file="$target.partial"
  # A dump that cannot be listed is not a backup.
  pg_restore --list "$target.partial" >/dev/null
  mv "$target.partial" "$target"
  log "wrote $(basename "$target") ($(du -h "$target" | cut -f1))"
  # Keep the newest BACKUP_KEEP, delete the rest.
  ls -1t "$BACKUP_DIR"/terror-*.dump 2>/dev/null | tail -n +"$((BACKUP_KEEP + 1))" | while read -r old; do
    rm -f "$old"; log "pruned $(basename "$old")"
  done
  rm -f "$BACKUP_DIR"/*.partial
  upload "$target" || log "UPLOAD FAILED: the dump exists on this server only"
}

verify() {
  newest=$(ls -1t "$BACKUP_DIR"/terror-*.dump 2>/dev/null | head -1)
  [ -n "$newest" ] || { log "no dump to verify"; exit 1; }
  scratch="restore_check_$$"
  log "restoring $(basename "$newest") into scratch database $scratch"
  createdb "$scratch"
  trap 'dropdb --if-exists "$scratch"' EXIT
  pg_restore --no-owner --dbname="$scratch" "$newest"
  status=0
  for table in $(psql -Atc "select tablename from pg_tables where schemaname='public' order by 1"); do
    live=$(psql -Atc "select count(*) from \"$table\"")
    restored=$(psql -d "$scratch" -Atc "select count(*) from \"$table\"" 2>/dev/null || echo MISSING)
    echo "  $table: live=$live restored=$restored"
    [ "$restored" != "MISSING" ] || status=1
  done
  if [ "$status" = 0 ]; then log "restore check passed: every table restored"; else log "RESTORE CHECK FAILED"; fi
  return "$status"
}

case "${1:-loop}" in
  once) once ;;
  verify) verify ;;
  loop)
    log "started; daily at ${BACKUP_HOUR_UTC}:00 UTC, keeping $BACKUP_KEEP in $BACKUP_DIR"
    # One at start-up if there is none from today, so a fresh deploy is
    # never a day without a backup.
    ls "$BACKUP_DIR"/terror-"$(date -u +%Y%m%d)"-*.dump >/dev/null 2>&1 || once || log "BACKUP FAILED"
    while true; do
      now=$(date -u +%s)
      next=$(( (now / 86400) * 86400 + BACKUP_HOUR_UTC * 3600 ))
      [ "$next" -gt "$now" ] || next=$(( next + 86400 ))
      sleep $(( next - now ))
      once || log "BACKUP FAILED"
    done ;;
  *) echo "usage: backup.sh [loop|once|verify]" >&2; exit 2 ;;
esac
