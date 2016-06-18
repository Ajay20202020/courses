/************************************************************************************
 * MainSerch.java
 *
 * Created on 18 November 2015
 *
 * Query search:  'food NOT drink' (in city pages)
 * 
 * References
 * http://oak.cs.ucla.edu/cs144/projects/lucene/
 * http://twiki.di.uniroma1.it/pub/Estrinfo/WebHome/lucene.pdf
 * http://lucene.apache.org/core/2_9_4/queryparsersyntax.html#+
 * http://www.avajava.com/tutorials/lessons/how-do-i-combine-queries-with-a-boolean-query.html
 * http://www.avajava.com/tutorials/lessons/how-do-i-query-for-words-near-each-other-with-a-phrase-query.html
 * http://stackoverflow.com/questions/15226337/why-does-lucene-queryparser-needs-an-analyzer
 * http://www.avajava.com/tutorials/lessons/how-do-i-combine-queries-with-a-boolean-query.html
 * http://www.avajava.com/tutorials/lessons/how-do-i-query-for-words-near-each-other-with-a-phrase-query.html
 * http://www.tutorialspoint.com/lucene/lucene_fuzzyquery.htm
 *************************************************************************************/

import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.IndexSearcher;


/**
 *
 * @author John
 */
public class MainSearch {
    
    /** Creates a new instance of Main */
    public MainSearch() {
    }
    
    /**
     * @param args the command line arguments
     * @throws IOException 
     * @throws ParseException 
     */
    public static void main(String[] args) throws IOException, ParseException {
    	
    	int nDocs = 250; // all
    	//String query = "+city_text:china";
    	String query = "food NOT drink";
    	//String query = "+Greek +Roman -Persian";
    	String output_file_name = "query_interesting.txt";
    	
    	
        // perform search on "Notre Dame museum"
        // and retrieve the top 100 result
    	
    	PrintStream ps = new PrintStream(output_file_name);
		PrintStream orig = System.out;
		System.setOut(ps);	
		
        SearchEngine se = new SearchEngine();
        TopDocs topDocs = se.performSearch(query, nDocs);

        System.out.println("Results found: " + topDocs.totalHits);
        ScoreDoc[] hits = topDocs.scoreDocs;
        for (int i = 0; i < hits.length; i++) {
            Document doc = se.getDocument(hits[i].doc);
            System.out.println(doc.get("city_name")
                               + ", " + doc.get("country_name")
                               + " (" + hits[i].score + ")");

        }
        
        //Functions.writeFile(String output_file_name, 
        
        System.setOut(orig);
		ps.close();
        		
        System.out.println("performSearch done");

    }

    

}
