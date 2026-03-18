/**
 * Centralized logging with context for debugging.
 */

export function logInfo(message: string, context?: Record<string, unknown>): void {
  const ctx = context ? ` | ${JSON.stringify(context)}` : '';
  console.log(`[INFO] ${message}${ctx}`);
}

export function logError(message: string, context?: Record<string, unknown>): void {
  const ctx = context ? ` | ${JSON.stringify(context)}` : '';
  console.error(`[ERROR] ${message}${ctx}`);
}

export function logWarn(message: string, context?: Record<string, unknown>): void {
  const ctx = context ? ` | ${JSON.stringify(context)}` : '';
  console.warn(`[WARN] ${message}${ctx}`);
}
