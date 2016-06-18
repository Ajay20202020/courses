/*
 * Main.java
 *
 * Created on 18 November 2015
 * http://oak.cs.ucla.edu/cs144/projects/lucene/
 */

import java.util.Iterator;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.IndexWriter;

import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.IndexSearcher;


/**
 *
 * @author Steven
 */
public class MainCreateIndex {
    
    /** Creates a new instance of Main */
    public MainCreateIndex() {
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
    	
    	String jsonFileName = "webscraped_data.json";

      try {
	// build a lucene index
        System.out.println("rebuildIndexes");
        Indexer  indexer = new Indexer();
        indexer.rebuildIndexes(jsonFileName);
        System.out.println("rebuildIndexes done");
        
      } catch (Exception e) {
        System.out.println("Exception caught.\n");
      }
    }
    
}
