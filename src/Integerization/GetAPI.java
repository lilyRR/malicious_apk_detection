package cqu.van;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;


import soot.MethodOrMethodContext;
import soot.Scene;
import soot.SootMethod;
import soot.jimple.infoflow.android.SetupApplication;
import soot.jimple.toolkits.callgraph.CallGraph;
import soot.jimple.toolkits.callgraph.Targets;

public class GetAPI {
    //设置android的jar包目录
    public final static String androidPlatformPath = "F:\\Android Q\\android-platform";
    //设置要分析的APK文件
    public final static String appDirPath = "F:\\Android Q\\API";
    //设置结果输出的路径

    //public final static String AnaPath = "F:\\Sample\\malapi";

    public final static String SenapiPath = "";
    static Object ob = new Object();

    private static Map<String,Boolean> visited = new HashMap<String,Boolean>();
    private static  Vector<String> file_vec =  new Vector<String>();
    private static  Vector<String> sen_api =  new Vector<String>();


    public static void getfile(){
        File file = new File(appDirPath);
        File[] tempList = file.listFiles();
        for (int i = 0; i < tempList.length; i++) {
            if (tempList[i].isFile()) {
                int lenname =tempList[i].getName().length();
                if(tempList[i].getName().substring(lenname-4).equals(".apk"))
                {
                    file_vec.addElement(tempList[i].getName());
                }

            }
        }

    }

    //apk路径，遍历模式(0只给出边，1给出所有路径)，输出位置
    public static void getapi(String appname,int mode,String AnaPath) {
        SetupApplication app = new SetupApplication(androidPlatformPath,appDirPath+File.separator+appname);
        soot.G.reset();
        //传入AndroidCallbacks文件
        app.setCallbackFile(CGGenerator.class.getResource("AndroidCallbacks.txt").getFile());
        //构造调用图，但是不进行数据流分析
        app.constructCallgraph();

        //SootMethod 获取函数调用图
        SootMethod entryPoint = app.getDummyMainMethod();
        CallGraph cg = Scene.v().getCallGraph();

        // System.out.println(apppath.substring(0,apppath.length()-4)+".txt");
        File oFile = new File(AnaPath+File.separator+appname.substring(0,appname.length()-4)+".txt");
        //可视化函数调用图

        switch (mode) {
            case 0:
                visit(cg,entryPoint,oFile,entryPoint.getSignature());
                break;
            case 1:
                visit1(cg,entryPoint,oFile,entryPoint.getSignature());
                break;

            default:
                break;
        }



    }

    private static void visit(CallGraph cg,SootMethod m,File oFile,String pString){
        //在soot中，函数的signature就是由该函数的类名，函数名，参数类型，以及返回值类型组成的字符串
        String identifier = m.getSignature();
        //记录是否已经处理过该点
        visited.put(identifier, true);
        //以函数的signature为label在图中添加该节点
        //获取该函数调用的函数
        Iterator<MethodOrMethodContext> ctargets = new Targets(cg.edgesOutOf(m));
        if(ctargets != null){
            while(ctargets.hasNext())
            {
                SootMethod c = (SootMethod) ctargets.next();
                if(c == null){
                    System.out.println("c is null");
                }
                //将被调用的函数加入图中

                //添加一条指向该被调函数的边
                //pString = pString+"-->"+ c.getSignature();
                writerow(oFile,delpar(identifier)+"-->"+ delpar(c.getSignature()));
                if(!visited.containsKey(c.getSignature())){
                    //递归
                    visit(cg,c,oFile,pString);
                }
                //writerow(oFile,pString);
            }
        }
    }


    private static void visit1(CallGraph cg,SootMethod m,File oFile,String pString){
        //在soot中，函数的signature就是由该函数的类名，函数名，参数类型，以及返回值类型组成的字符串
        String identifier = m.getSignature();
        //记录是否已经处理过该点
        visited.put(identifier, true);
        //以函数的signature为label在图中添加该节点
        //获取该函数调用的函数
        Iterator<MethodOrMethodContext> ctargets = new Targets(cg.edgesOutOf(m));
        if(ctargets != null){
            while(ctargets.hasNext())
            {
                SootMethod c = (SootMethod) ctargets.next();
                if(c == null){
                    System.out.println("c is null");
                }
                //将被调用的函数加入图中

                //添加一条指向该被调函数的边
                pString = pString+"-->"+ c.getSignature();
                //writerow(oFile,delpar(identifier)+"-->"+ delpar(c.getSignature()));
                if(!visited.containsKey(c.getSignature())){
                    //递归
                    visit1(cg,c,oFile,pString);
                }
                writerow(oFile,pString);
            }
        }
    }


    private static void visit2(CallGraph cg,SootMethod m,File oFile,String pString){
        //在soot中，函数的signature就是由该函数的类名，函数名，参数类型，以及返回值类型组成的字符串
        String identifier = m.getSignature();
        //记录是否已经处理过该点
        visited.put(identifier, true);
        //以函数的signature为label在图中添加该节点
        //获取该函数调用的函数
        Iterator<MethodOrMethodContext> ctargets = new Targets(cg.edgesOutOf(m));
        if(ctargets != null){
            while(ctargets.hasNext())
            {
                SootMethod c = (SootMethod) ctargets.next();
                if(c == null){
                    System.out.println("c is null");
                }
                //将被调用的函数加入图中

                if(!visited.containsKey(c.getSignature())){
                    //递归
                    writerow(oFile,delpar(c.getSignature()));
                    visit2(cg,c,oFile,pString);
                }
            }
        }
    }




    private static void writerow (File ofile, String s) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(ofile,true);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        PrintWriter pw = new PrintWriter(fw);
        pw.println(s);
        pw.flush();
        try {
            fw.flush();
            pw.close();
            fw.close();
        } catch (Exception e) {
            // TODO: handle exception
        }

    }


    public static boolean judgesen(String api)
    {
        return true;
    }
    public static String delpar(String s)
    {
        int x = s.indexOf('(');
        int y = s.indexOf(')');
        String b = s.substring(0,x+1);
        String e = s.substring(y,s.length());
        s=b+e;
        x = s.indexOf(':');
        String r = s.substring(x+1,s.length()-1);
        return r;
    }

    public static String cutMethod(String s)
    {
        int x = s.indexOf(':');
        String e = s.substring(x+1,s.length()-1);
        return e;
    }

    public static void main(String[] args){
        // System.out.print(delpar("<org.json.JSONObject: java.lang.String getString(java.lang.String)>"));

        File LOG = new File("F:\\Android Q\\API\\log.txt");
        getfile();
//		  for(int i=0;i<file_vec.size();i++)
//		  {
//			  System.out.println(file_vec.elementAt(i));
//		  }
        for(int i=0;i<file_vec.size();i++)
        {
            try {
                getapi(file_vec.elementAt(i),0,"F:\\Android Q\\API");

            } catch (Exception e) {
                // TODO: handle exception
                writerow(LOG, file_vec.elementAt(i));
            }

            System.out.println(i);
            //提取api序列
        }
    }


}

