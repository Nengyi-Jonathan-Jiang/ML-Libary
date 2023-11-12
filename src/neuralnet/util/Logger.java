package neuralnet.util;

public class Logger {
    public enum Level {
        PLAIN, VERBOSE, DEBUG
    }

    private static Level level = Level.PLAIN;
    public static void setLevel(Level level) { Logger.level = level; }

    public static void logDebug(String s) {
        switch (level) {
            case DEBUG -> Logger.logDebug(s);
            default -> {}
        }
    }
}
