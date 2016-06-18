/************************************************************************************
 * SearchEnginejava
 *
 * Created on 18 November 2015
 *
 * Search engine using searcher and query parser (default: city pages)
 * 
 * References
 * http://oak.cs.ucla.edu/cs144/projects/lucene/
 * http://stackoverflow.com/questions/2005084/how-to-specify-two-fields-in-lucene-queryparser
 * http://lucene.apache.org/core/2_9_4/queryparsersyntax.html#+
 * http://www.avajava.com/tutorials/lessons/how-do-i-combine-queries-with-a-boolean-query.html
 * http://www.avajava.com/tutorials/lessons/how-do-i-query-for-words-near-each-other-with-a-phrase-query.html
 * http://stackoverflow.com/questions/15226337/why-does-lucene-queryparser-needs-an-analyzer
 * http://www.avajava.com/tutorials/lessons/how-do-i-combine-queries-with-a-boolean-query.html
 * http://www.avajava.com/tutorials/lessons/how-do-i-query-for-words-near-each-other-with-a-phrase-query.html
 * http://www.tutorialspoint.com/lucene/lucene_fuzzyquery.htm
 *************************************************************************************/

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.File;

import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;


public class SearchEngine {
    private IndexSearcher searcher = null;
    private QueryParser parser = null;
    
    /** Creates a new instance of SearchEngine */
    public SearchEngine() throws IOException {
    	
    	String DEFAULT_FIELD = "city_text";
        searcher = new IndexSearcher(DirectoryReader.open(FSDirectory.open(new File("index-directory"))));
        parser = new QueryParser(DEFAULT_FIELD , new EnglishAnalyzer());
        //This default field is used if the query string does not specify the search field.
    }
    
    public TopDocs performSearch(String queryString, int n)
    throws IOException, ParseException {
        Query query = parser.parse(queryString); 
        displayQuery(query);
        return searcher.search(query, n);
    }

    public Document getDocument(int docId)
    throws IOException {
        return searcher.doc(docId);
    }
    
    public static void displayQuery(Query query) {
		System.out.println("Query: " + query.toString());
	}
}
