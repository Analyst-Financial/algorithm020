package algorithm;

/**
 * @ClassName Notes
 * @Description
 * @Author Administrator
 * @Date 2020/11/30  0:19
 * @Version 1.0
 **/
public class Notes {

    // 推荐一个小网站给大家，可以看各种算法的显示图：https://visualgo.net
    // 比如 https://visualgo.net/en/recursion 这里可以看递归。 Fib的递归状态树如图：
    // 发现递归，分治和动规都差别不大，本质上就是寻找有规律的重复性。

    /**
     * Java 平台提供了两种类型的字符串：String 和StringBuffer/StringBuilder，
     * 它们可以储存和操作字符串。其中String 是只读字符串，也就意味着String
     * 引用的字符串内容是不能被改变的。而StringBuffer/StringBuilder 类表示的
     * 字符串对象可以直接进行修改。StringBuilder 是Java 5 中引入的，它和
     * StringBuffer 的方法完全相同，区别在于它是在单线程环境下使用的，因为它
     * 的所有方面都没有被synchronized 修饰，因此它的效率也比StringBuffer 要高。
     * 面试题1 - 什么情况下用+运算符进行字符串连接比调用
     * StringBuffer/StringBuilder 对象的append 方法连接字符串性能更好？
     * 面试题2 - 请说出下面程序的输出。
     * *****************************************************************************************************************
     * 补充：解答上面的面试题需要清除两点：
     * (1)String 对象的intern 方法会得到字符串对象在常量池中对应的版本的引
     * 用（如果常量池中有一个字符串与String 对象的equals 结果是true），如
     * 果常量池中没有对应的字符串，则该字符串将被添加到常量池中，然后返回常量
     * 池中字符串的引用；
     * (2)字符串的+操作其本质是创建了StringBuilder 对象进行append 操作，然
     * 后将拼接后的StringBuilder 对象用toString 方法处理成String 对象，这
     * 一点可以用javap -c StringEqualTest.class 命令获得class 文件对应的
     * JVM 字节码指令就可以看出来。
     * @param args
     */
    public static void main(String[] args) {
        // 19、String 和StringBuilder、StringBuffer 的区 别？
        String s1 = "Programming";
        String s2 = new String("Programming");
        String s3 = "Program";
        String s4 = "ming";
        String s5 = "Program" + "ming";
        String s6 = s3 + s4;
        System.out.println(s1 == s2);//false
        System.out.println(s1 == s5);//true
        System.out.println(s1 == s6);//false
        System.out.println(s1 == s6.intern());//true
        System.out.println(s2 == s2.intern());//false
    }

}
