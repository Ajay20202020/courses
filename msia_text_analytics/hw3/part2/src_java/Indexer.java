/************************************************************************************
 * Indexer.java
 *
 * Created on 18 November 2015
 *
 * Index Class
 * Creates the indices based on json file
 * 
 * Use method rebuildIndexes(String jsonFileName) to build index
 * 
 * References
 * http://oak.cs.ucla.edu/cs144/projects/lucene/
 *************************************************************************************/


import java.io.IOException;
import java.io.StringReader;
import java.util.Iterator;
import java.io.File;
import java.io.FileReader;

import java.io.FileReader;
import java.util.Iterator;
 
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField; 
import org.apache.lucene.document.TextField; 
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;


public class Indexer {
    
    /** Creates a new instance of Indexer */
    public Indexer() {
    }
 
    private IndexWriter indexWriter = null;
    
    public IndexWriter getIndexWriter(boolean create) throws IOException {
        if (indexWriter == null) {
            Directory indexDir = FSDirectory.open(new File("index-directory"));
            IndexWriterConfig config = new IndexWriterConfig(Version.LUCENE_4_10_2, new EnglishAnalyzer());
            indexWriter = new IndexWriter(indexDir, config);
        }
        return indexWriter;
   }    
   
    public void closeIndexWriter() throws IOException {
        if (indexWriter != null) {
            indexWriter.close();
        }
   }
    
    public void indexJSON (JSONObject jsonObject, String key) throws IOException {
    	/**
    	 * indexJSON
    	 * 
    	 * Create a document for the JSON file
    	 * @param (JSONObject) jsonObject
    	 * @param String key
    	 * @throws IOException
    	 */
    	
    	JSONObject description = (JSONObject)jsonObject.get(key);
    	
    	String description_city = (String) description.get("city");
    	String description_country = (String) description.get("country");
    	
		String[] parts = key.split("_");
		
        System.out.println("Indexing City, Country: " + parts[0] + ", " + parts[1]);
        IndexWriter writer = getIndexWriter(false);
        Document doc = new Document();
        
        doc.add(new StringField("city_name", parts[0], Field.Store.YES));
        doc.add(new StringField("country_name", parts[1], Field.Store.YES));
        //doc.add(new TextField("city_text", description_city, Field.Store.NO));
        //doc.add(new TextField("country_text", description_country, Field.Store.NO));
        
        FieldType type = new FieldType();
        type.setIndexed(true);
        type.setStored(true);
        type.setStoreTermVectors(true); //TermVectors are needed for MoreLikeThi
        
        doc.add(new Field("city_text", description_city, type));
        doc.add(new Field("country_text", description_country, type));
        
        
        writer.addDocument(doc);
    }   
    
    public void rebuildIndexes(String jsonFileName) throws IOException {
    	/**
    	 * rebuildIndexex
    	 * 
    	 * Creates the indices
    	 * @param String jsonFileName
    	 * @throws IOException
    	 */
    	
          //
          // Erase existing index
          //
          getIndexWriter(true);
          //
          // Index all entries
          //
          JSONParser parser = new JSONParser();
          try {
        	  Object obj = parser.parse(new FileReader(jsonFileName));
        	  
        	  JSONObject jsonObject = (JSONObject) obj;
              
              for(Iterator iterator = jsonObject.keySet().iterator(); iterator.hasNext();) {
                  String key = (String) iterator.next();
                  indexJSON(jsonObject, key);
              }
  
          } catch (Exception e) {
              e.printStackTrace();
          }
              //
          // Don't forget to close the index writer when done
          //
          closeIndexWriter();
     }    
}
