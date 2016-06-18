/************************************************************************************
 * MainSerchBoolean.java
 *
 * Created on 18 November 2015
 *
 * Boolean search: Green and Roman but not persian (in city pages)
 * 
 * References
 * http://oak.cs.ucla.edu/cs144/projects/lucene/
 * http://lucene.apache.org/core/2_9_4/queryparsersyntax.html#+
 * http://www.avajava.com/tutorials/lessons/how-do-i-combine-queries-with-a-boolean-query.html
 * http://www.avajava.com/tutorials/lessons/how-do-i-query-for-words-near-each-other-with-a-phrase-query.html
 * http://stackoverflow.com/questions/15226337/why-does-lucene-queryparser-needs-an-analyzer
 * http://www.avajava.com/tutorials/lessons/how-do-i-combine-queries-with-a-boolean-query.html
 * http://www.avajava.com/tutorials/lessons/how-do-i-query-for-words-near-each-other-with-a-phrase-query.html
 * http://www.tutorialspoint.com/lucene/lucene_fuzzyquery.htm
 *************************************************************************************/

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.QueryBuilder;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.IndexSearcher;

import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.TermQuery;


public class MainSearchBoolean {
    
    /** Creates a new instance of Main */
    public MainSearchBoolean() {
    }
    
    /**
     * @param args the command line arguments
     * @throws IOException 
     * @throws ParseException 
     */
    public static void main(String[] args) throws IOException, ParseException {
    	
   
    	String output_file_name = "query_boolean.txt";
        
        IndexSearcher is = new IndexSearcher(DirectoryReader.open(FSDirectory.open(new File("index-directory"))));
        
        String FIELD_CONTENTS = "city_text";
        int nDocs = 249;
        
        QueryParser parser = new QueryParser(FIELD_CONTENTS , new EnglishAnalyzer());
        BooleanQuery booleanQuery = new BooleanQuery();
        
        String query1s = "Greek";
        String query2s = "Roman";
        String query3s = "Persian";
        
        // This is if use the standard analyzer
        //Query query1 = new TermQuery(new Term(FIELD_CONTENTS, query1s));
		//Query query2 = new TermQuery(new Term(FIELD_CONTENTS, query2s));
		//Query query3 = new TermQuery(new Term(FIELD_CONTENTS, query3s));

		//booleanQuery.add(query1, BooleanClause.Occur.MUST);
		//booleanQuery.add(query2, BooleanClause.Occur.MUST);
		//booleanQuery.add(query3, BooleanClause.Occur.MUST_NOT);
		
		// This is for english analyzer
		booleanQuery.add(parser.parse(query1s), BooleanClause.Occur.MUST);
		booleanQuery.add(parser.parse(query2s), BooleanClause.Occur.MUST);
		booleanQuery.add(parser.parse(query3s), BooleanClause.Occur.MUST_NOT);
		
		PrintStream ps = new PrintStream(output_file_name);
		PrintStream orig = System.out;
		System.setOut(ps);	
		
		displayQuery(booleanQuery);	
		
		// Equivalent to  Query: +city_text:greek +city_text:roman -city_text:persian
		TopDocs topDocs = is.search(booleanQuery,nDocs);
		
		//topDocs = se.performSearch(booleanQuery, nDocs);
		//displayHits(hits);
		
		/*
		QueryBuilder builder = new QueryBuilder(new StandardAnalyzer());
		Query a = builder.createBooleanQuery(FIELD_CONTENTS, "Greek",BooleanClause.Occur.MUST);
		*/
		
			

        System.out.println("Results found: " + topDocs.totalHits);
        ScoreDoc[] hits = topDocs.scoreDocs;
        for (int i = 0; i < hits.length; i++) {
            Document doc = is.doc(hits[i].doc);
            System.out.println(doc.get("city_name")
                               + ", " + doc.get("country_name")
                               + " (" + hits[i].score + ")");

        }
        		
        System.setOut(orig);
		ps.close();
		
        System.out.println("performSearch done");
        
    }
    
    public static void displayQuery(Query query) {
		System.out.println("Query: " + query.toString());
	}
    
    /*
    public static void displayHits(Hits hits) throws CorruptIndexException, IOException {
		System.out.println("Number of hits: " + hits.length());

		Iterator<Hit> it = hits.iterator();
		while (it.hasNext()) {
			Hit hit = it.next();
			Document document = hit.getDocument();
			String path = document.get(FIELD_PATH);
			System.out.println("Hit: " + path);
		}
		System.out.println();
	}
    */
    
}
