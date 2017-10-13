//
//  Detail.swift
//  luminate
//
//  Created by Prayash Thapa on 10/7/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import UIKit
import LBTAComponents

class DetailView: UIView {
    var textView = UITextView()
    var nameView = UITextView()
    var button = UIButton()
    
    override init (frame: CGRect) {
        super.init(frame: frame)
        self.backgroundColor = UIColor(red: 0.08, green: 0.5, blue: 0.78, alpha: 0.3)
        self.translatesAutoresizingMaskIntoConstraints = false
        
        setupGradient()
        setupTextView()
        setupNameView()
        setupCloseButton()
    }
    
    convenience init() {
        self.init(frame: CGRect.zero)
    }
    
    func setupGradient() {
        let imageView: UIImageView = {
            let view = UIImageView()
            view.image = UIImage(named: "gradient")
            return view
        }()
        
        addSubview(imageView)
        imageView.anchor(self.topAnchor, left: self.leftAnchor, bottom: self.bottomAnchor, right: self.rightAnchor, topConstant: 0, leftConstant: 0, bottomConstant: 0, rightConstant: 0, widthConstant: 0, heightConstant: 0)
    }
    
    func setupCloseButton() {
        let button: UIButton = {
            let view = UIButton()
            view.setImage(UIImage(named: "CloseIcon"), for: .normal)
            return view
        }()
        
         button.addTarget(self, action: #selector(hideSelf), for: .touchUpInside)
        
        addSubview(button)
        button.anchor(self.topAnchor, left: self.leftAnchor, bottom: nil, right: nil, topConstant: 12, leftConstant: 12, bottomConstant: 0, rightConstant: 0, widthConstant: 18, heightConstant: 18)
    }
    
    func setupTextView() {
        textView.translatesAutoresizingMaskIntoConstraints = false
        textView.font = UIFont.boldSystemFont(ofSize: 24)
        textView.textColor = UIColor.white
        textView.backgroundColor = UIColor.clear
        textView.isEditable = false
        textView.textAlignment = .center
        self.addSubview(textView)
        
        textView.anchor(nil, left: self.leftAnchor, bottom: self.bottomAnchor, right: self.rightAnchor, topConstant: 0, leftConstant: 0, bottomConstant: 0, rightConstant: 0, widthConstant: 0, heightConstant: 250)
    }
    
    func setupNameView() {
        nameView.translatesAutoresizingMaskIntoConstraints = false
        nameView.font = UIFont.boldSystemFont(ofSize: 24)
        nameView.textColor = UIColor.white
        nameView.backgroundColor = UIColor.clear
        nameView.isEditable = false
        nameView.textAlignment = .center
        self.addSubview(nameView)
        
        nameView.anchor(nil, left: self.leftAnchor, bottom: self.bottomAnchor, right: self.rightAnchor, topConstant: 0, leftConstant: 0, bottomConstant: 0, rightConstant: 0, widthConstant: 0, heightConstant: 50)
    }
    
    @objc func hideSelf() {
        self.isHidden = true
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
