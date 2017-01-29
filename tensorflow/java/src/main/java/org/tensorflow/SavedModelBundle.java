package org.tensorflow;

import java.util.Set;

/**
 * SavedModel representation once the SavedModel is loaded from storage.
 */
public class SavedModelBundle implements AutoCloseable {

    private final Graph graph;

    private final Session session;

    private final byte[] metaGraphDef;

    public SavedModelBundle(Graph graph, Session session, byte[] metaGraphDef) {
        this.graph = graph;
        this.session = session;
        this.metaGraphDef = metaGraphDef;
    }

    public byte[] metaGraphDef() {
        return metaGraphDef;
    }

    public Graph graph() {
        return graph;
    }

    public Session session() {
        return session;
    }

    @Override
    public void close() throws Exception {
        session.close();
        graph.close();
    }

    public static SavedModelBundle fromHandle(long graphHandle, long sessionHandle, byte[] metaGraphDef) {
        Graph graph = new Graph(graphHandle);
        Session session = new Session(graph, sessionHandle);
        return new SavedModelBundle(graph, session, metaGraphDef);
    }

    public static SavedModelBundle loadSavedModel(String exportDir, Set<String> tags) {
        return load(exportDir, tags.toArray(new String[tags.size()]));
    }

    private static native SavedModelBundle load(String exportDir, String[] tags);

    static {
        TensorFlow.init();
    }
}
