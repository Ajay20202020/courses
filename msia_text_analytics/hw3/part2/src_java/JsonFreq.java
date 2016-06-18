/************************************************************************************
 * JsonFreq.java
 *
 * Created on 18 November 2015
 *
 * Outputs a json file with {"cityName_countryName": {"Term": frequency}
 * 
 * References
 * http://www.sergiy.ca/how-to-iterate-over-a-map-in-java/
 * http://doduck.com/lucene-morelikethis-java-example/
 * http://stackoverflow.com/questions/20899839/retreiving-values-from-nested-json-object
 * http://www.mkyong.com/java/how-to-get-the-current-working-directory-in-java/
 * http://www.mkyong.com/java/json-simple-example-read-and-write-json/
 *************************************************************************************/


import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.FileWriter;
import java.io.IOException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class JsonFreq {
	
	 public static void main(String[] args) throws IOException {
		
		 IndexReader reader = DirectoryReader.open(FSDirectory.open(new File("index-directory")));
		 
		 String CONTENT = "country_text";
		 String OUT_FILE_NAME = "vectorFreq.json";
		 
		 JSONObject obj = new JSONObject();
		 
		 for (int docId=0; docId < reader.maxDoc(); docId++) {
			 
			 Document doc = reader.document(docId);
			 String city_name = doc.get("city_name");
			 String country_name = doc.get("country_name");
			 String name = city_name + "_" + country_name;
			 
			 Terms vector = reader.getTermVector(docId, CONTENT);
			 JSONObject obj_freq = getJsonFreq(vector);
			 obj.put(name, obj_freq );

			 System.out.println("Document " + name + " loaded");
	
			 //String terms[] = vector.getTerms();
	         //int termCount = vector.length;
	         //int freqs[] = tfv.getTermFrequencies();
			 
			} // end for
		 
		 saveToJson(obj, OUT_FILE_NAME);
		 
	 } // end main
	 
	 public static JSONObject getJsonFreq(Terms vector) throws IOException{
		 /**
		 * saveToJson
		 * Function write to json file
		 * @param JSONObject obj
		 * @param String fileName 
		 * @throws IOException
		 */
		 
		 JSONObject obj_freq = new JSONObject();
		 
		 TermsEnum termsEnum = null;
		 termsEnum = vector.iterator(termsEnum);
		 //Map<String, Integer> frequencies = new HashMap<>();
		 BytesRef text = null;
		 while ((text = termsEnum.next()) != null) {
		     String term = text.utf8ToString();
		     int freq = (int) termsEnum.totalTermFreq();
		     //frequencies.put(term, freq);
		     //terms.add(term);
		     
		     obj_freq.put(term, freq);     
		     
		 } // end while
		 
		 /*
		 for (String key : frequencies.keySet()) {
			    System.out.println("Key = " + key + ", Frequency = " + frequencies.get(key));
			}
		 */	
		 
		 return obj_freq;	 
		 
	 } // end getJsonFreq
	 
	 
	 public static void saveToJson(JSONObject obj, String fileName) throws IOException {	 
		/**
		 * saveToJson
		 * Function write to json file
		 * @param JSONObject obj
		 * @param String fileName 
		 * @throws IOException
		 */
		try {

			FileWriter file = new FileWriter(fileName);
			file.write(obj.toJSONString());
			file.flush();
			file.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		 String workingDir = System.getProperty("user.dir");
		 System.out.println("File " + fileName + " in current working directory : " + workingDir);
		 
	 } // saveToJson 

} // 
